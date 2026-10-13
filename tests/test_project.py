import sys
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent import create_workflow
from assistant import DocumentAssistant
from langchain_core.messages import AIMessage, HumanMessage, message_to_dict
from retrieval import SimulatedRetriever
from schemas import (
    AnswerResponse,
    DocumentChunk,
    SessionState,
    SummarizationResponse,
    UpdateMemoryResponse,
    UserIntent,
)
from tools import (
    ToolLogger,
    create_calculator_tool,
    create_document_reader_tool,
    create_document_search_tool,
    create_document_statistics_tool,
)


class FakeStructuredModel:
    def __init__(self, schema, intent_type, document_ids=None):
        self.schema = schema
        self.intent_type = intent_type
        self.document_ids = document_ids or []

    def invoke(self, _prompt):
        if self.schema is UserIntent:
            return UserIntent(
                intent_type=self.intent_type,
                confidence=1.0,
                reasoning="Deterministic test intent",
            )
        if self.schema is UpdateMemoryResponse:
            return UpdateMemoryResponse(
                summary="Test conversation summary",
                document_ids=self.document_ids,
            )
        raise AssertionError(f"Unexpected structured output schema: {self.schema}")


class FakeChatModel:
    def __init__(self, intent_type, document_ids=None):
        self.intent_type = intent_type
        self.document_ids = document_ids or []

    def with_structured_output(self, schema):
        return FakeStructuredModel(schema, self.intent_type, self.document_ids)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.retriever = SimulatedRetriever()

    def test_loads_all_sample_documents(self):
        self.assertEqual(
            set(self.retriever.documents),
            {"INV-001", "INV-002", "INV-003", "CON-001", "CLM-001"},
        )
        self.assertEqual(len(self.retriever.retrieve_all()), 5)

    def test_searches_by_type_and_keyword(self):
        invoices = self.retriever.retrieve_by_type("invoice")
        self.assertEqual({item.doc_id for item in invoices}, {"INV-001", "INV-002", "INV-003"})

        matches = self.retriever.retrieve_by_keyword("Acme Corporation")
        self.assertIn("INV-001", {item.doc_id for item in matches})

    def test_filters_documents_by_amount(self):
        matches = self.retriever.retrieve_by_amount_range(min_amount=50000)
        self.assertEqual({item.doc_id for item in matches}, {"INV-002", "INV-003", "CON-001"})

    def test_reads_documents_and_reports_statistics(self):
        document = self.retriever.get_document_by_id("INV-002")
        self.assertIsNotNone(document)
        self.assertIn("Total Due: $69,300", document.content)
        self.assertIsNone(self.retriever.get_document_by_id("MISSING"))

        stats = self.retriever.get_statistics()
        self.assertEqual(stats["total_documents"], 5)
        self.assertEqual(stats["documents_with_amounts"], 4)
        self.assertEqual(stats["total_amount"], 466250)


class SchemaTests(unittest.TestCase):
    def test_collection_fields_have_independent_empty_defaults(self):
        first = DocumentChunk(doc_id="A", content="one")
        second = DocumentChunk(doc_id="B", content="two")
        first.metadata["tag"] = "first"
        self.assertEqual(second.metadata, {})

        first_answer = AnswerResponse(
            question="Q", answer="A", confidence=1.0
        )
        second_answer = AnswerResponse(
            question="Q", answer="A", confidence=1.0
        )
        self.assertEqual(first_answer.sources, [])
        self.assertEqual(second_answer.sources, [])
        self.assertIsNot(first_answer.sources, second_answer.sources)

        summary = SummarizationResponse(
            original_length=0,
            summary="",
            key_points=[],
        )
        session = SessionState(session_id="test", user_id="user")
        self.assertEqual(summary.document_ids, [])
        self.assertEqual(session.conversation_history, [])
        self.assertEqual(session.document_context, [])


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.logger = ToolLogger(logs_dir=self.temp_dir.name)
        self.retriever = SimulatedRetriever()

    def test_calculator_computes_and_logs(self):
        calculator = create_calculator_tool(self.logger)
        self.assertEqual(calculator.invoke({"expression": "2 + 3 * 4"}), "Result: 14")
        logs = self.logger.get_logs()
        self.assertEqual(logs[-1]["tool_name"], "calculator")
        self.assertEqual(logs[-1]["input"]["expression"], "2 + 3 * 4")

    def test_calculator_rejects_non_math_input(self):
        calculator = create_calculator_tool(self.logger)
        result = calculator.invoke({"expression": "__import__('os').system('whoami')"})
        self.assertTrue(result.startswith("Error:"))
        self.assertEqual(len(self.logger.get_logs()), 1)

    def test_search_all_returns_every_document(self):
        search = create_document_search_tool(self.retriever, self.logger)
        result = search.invoke({"query": "", "search_type": "all"})
        for doc_id in self.retriever.documents:
            self.assertIn(doc_id, result)

    def test_reader_and_statistics_tools_return_results(self):
        reader = create_document_reader_tool(self.retriever, self.logger)
        self.assertIn("Total Due: $69,300", reader.invoke({"doc_id": "INV-002"}))
        self.assertIn("not found", reader.invoke({"doc_id": "MISSING"}).lower())

        statistics = create_document_statistics_tool(self.retriever, self.logger)
        result = statistics.invoke({})
        self.assertIn("Total Documents: 5", result)
        self.assertIn("Total Amount: $466,250.00", result)


class WorkflowTests(unittest.TestCase):
    def test_routes_each_intent_through_agent_and_memory(self):
        for intent_type, expected_agent in (
            ("qa", "qa_agent"),
            ("summarization", "summarization_agent"),
            ("calculation", "calculation_agent"),
        ):
            with self.subTest(intent_type=intent_type):
                llm = FakeChatModel(intent_type, ["INV-001"])

                def fake_invoke_react_agent(schema, _messages, _llm, _tools):
                    return {"messages": [AIMessage(content="Agent response")]}, ["test_tool"]

                with patch("agent.invoke_react_agent", side_effect=fake_invoke_react_agent):
                    app = create_workflow(llm, [])
                    result = app.invoke(
                        {
                            "user_input": "Test request",
                            "messages": [],
                            "intent": None,
                            "next_step": "classify_intent",
                            "conversation_summary": "",
                            "active_documents": [],
                            "current_response": None,
                            "tools_used": [],
                            "session_id": "test-session",
                            "user_id": "test-user",
                            "actions_taken": [],
                        },
                        config={
                            "configurable": {
                                "thread_id": f"test-{intent_type}",
                                "llm": llm,
                                "tools": [],
                            }
                        },
                    )

                self.assertEqual(result["intent"].intent_type, intent_type)
                self.assertEqual(result["conversation_summary"], "Test conversation summary")
                self.assertEqual(result["active_documents"], ["INV-001"])
                self.assertEqual(result["next_step"], "end")
                self.assertEqual(result["actions_taken"], ["classify_intent", expected_agent])
                self.assertEqual(result["tools_used"], ["test_tool"])

    def test_question_regression_dataset(self):
        cases_path = Path(__file__).parent / "data" / "question_regression_cases.json"
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        retriever = SimulatedRetriever()

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ToolLogger(logs_dir=temp_dir)
            reader = create_document_reader_tool(retriever, logger)
            calculator = create_calculator_tool(logger)

            for index, case in enumerate(cases):
                with self.subTest(case=case["name"]):
                    llm = FakeChatModel(case["intent"], case["document_ids"])

                    def run_case_agent(schema, _messages, _llm, _tools):
                        evidence = [
                            reader.invoke({"doc_id": doc_id})
                            for doc_id in case["document_ids"]
                        ]
                        tool_names = (
                            ["document_reader"] if case["document_ids"] else []
                        )
                        if case["expression"]:
                            evidence.append(
                                calculator.invoke({"expression": case["expression"]})
                            )
                            tool_names.append("calculator")
                        response = "\n".join(evidence) or "No document lookup required."
                        return {"messages": [AIMessage(content=response)]}, tool_names

                    with patch("agent.invoke_react_agent", side_effect=run_case_agent):
                        app = create_workflow(llm, [])
                        result = app.invoke(
                            {
                                "user_input": case["question"],
                                "messages": [],
                                "intent": None,
                                "next_step": "classify_intent",
                                "conversation_summary": "",
                                "active_documents": [],
                                "current_response": None,
                                "tools_used": [],
                                "session_id": "regression-session",
                                "user_id": "test-user",
                                "actions_taken": [],
                            },
                            config={
                                "configurable": {
                                    "thread_id": f"question-case-{index}",
                                    "llm": llm,
                                    "tools": [],
                                }
                            },
                        )

                    self.assertEqual(result["intent"].intent_type, case["intent"])
                    self.assertEqual(result["active_documents"], case["document_ids"])
                    self.assertEqual(result["tools_used"], case["expected_tools"])
                    response = result["messages"][-1].content
                    for expected_text in case["expected_text"]:
                        self.assertIn(expected_text, response)
                    if case["expected_result"] is not None:
                        self.assertIn(str(case["expected_result"]), response)


class AssistantTests(unittest.TestCase):
    def test_processing_without_a_session_raises_clear_error(self):
        assistant = DocumentAssistant.__new__(DocumentAssistant)
        assistant.current_session = None

        with self.assertRaisesRegex(ValueError, "No active session"):
            assistant.process_message("Hello")

    def test_session_save_resume_restores_messages_summary_and_documents(self):
        class EchoWorkflow:
            def __init__(self):
                self.received_states = []

            def invoke(self, state, config):
                self.received_states.append((state, config))
                messages = [
                    *state["messages"],
                    HumanMessage(content=state["user_input"]),
                    AIMessage(content=f"Reply to: {state['user_input']}"),
                ]
                return {
                    **state,
                    "messages": messages,
                    "intent": UserIntent(
                        intent_type="qa",
                        confidence=1.0,
                        reasoning="Session persistence test",
                    ),
                    "conversation_summary": f"Summary after {state['user_input']}",
                    "active_documents": ["INV-002"],
                    "actions_taken": ["classify_intent", "qa_agent"],
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            first_assistant = DocumentAssistant.__new__(DocumentAssistant)
            first_assistant.session_storage_path = temp_dir
            first_assistant.llm = object()
            first_assistant.tools = []
            first_assistant.current_session = SessionState(
                session_id="resume-test",
                user_id="test-user",
            )
            first_assistant.workflow = EchoWorkflow()

            with redirect_stdout(StringIO()):
                first_assistant.start_session("test-user", "resume-test")
            first_result = first_assistant.process_message("First question")

            self.assertTrue(first_result["success"], first_result.get("error"))
            session_file = Path(temp_dir) / "resume-test.json"
            saved_data = json.loads(session_file.read_text(encoding="utf-8"))
            self.assertEqual(len(saved_data["conversation_history"]), 1)
            self.assertIsInstance(
                saved_data["conversation_history"][0]["messages"][0],
                dict,
            )

            restarted_assistant = DocumentAssistant.__new__(DocumentAssistant)
            restarted_assistant.session_storage_path = temp_dir
            restarted_assistant.llm = object()
            restarted_assistant.tools = []
            restarted_assistant.current_session = None
            restarted_assistant.workflow = EchoWorkflow()
            with redirect_stdout(StringIO()):
                restarted_assistant.start_session("test-user", "resume-test")

            second_result = restarted_assistant.process_message("Follow-up question")
            second_input, second_config = restarted_assistant.workflow.received_states[0]

            self.assertTrue(second_result["success"], second_result.get("error"))
            self.assertEqual(second_config["configurable"]["thread_id"], "resume-test")
            self.assertEqual(
                [message.content for message in second_input["messages"]],
                ["First question", "Reply to: First question"],
            )
            self.assertEqual(
                second_input["conversation_summary"],
                "Summary after First question",
            )
            self.assertEqual(second_input["active_documents"], ["INV-002"])


if __name__ == "__main__":
    unittest.main()
