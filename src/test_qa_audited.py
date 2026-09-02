"""Offline self-tests. No real API calls, model downloads, or legacy writes."""
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import qa_audit_runtime as runtime
import run_qa_audited as runner


class Clock:
    def __init__(self):
        self.now = 0.0
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


def response(content, *, finish="stop", usage=True):
    return types.SimpleNamespace(id="mock-response", model="mock-server-model", system_fingerprint="mock",
        choices=[types.SimpleNamespace(finish_reason=finish, message=types.SimpleNamespace(content=content))],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15} if usage else None)


VALID = {"answer": "42", "confidence": 0.7, "abstain": False, "citations": [], "reason": "mock only"}


def validate(value):
    if not isinstance(value, dict) or set(value) != set(VALID):
        raise runtime.SchemaError("mock schema rejection")
    return value


class MockClient:
    def __init__(self, outcomes, clock):
        self.outcomes, self.clock, self.calls = list(outcomes), clock, []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
        self.closed = False
    def create(self, **kwargs):
        self.calls.append(kwargs)
        self.clock.now += 0.25
        next_value = self.outcomes.pop(0)
        if isinstance(next_value, Exception):
            raise next_value
        return next_value
    def close(self):
        self.closed = True


class ApiError(Exception):
    def __init__(self, status):
        super().__init__("SECRET_MUST_NOT_BE_LOGGED")
        self.status_code = status


class RuntimeTests(unittest.TestCase):
    def run_request(self, outcomes):
        clock = Clock()
        client = MockClient(outcomes, clock)
        events = []
        request = runtime.request_payload("mock", [{"role": "user", "content": "question"}])
        result = runtime.execute_question(client, request, validate, events.append,
                                          clock=clock, sleep=clock.sleep)
        return result, client, events

    def test_success_and_exact_parameters(self):
        result, client, events = self.run_request([response(json.dumps(VALID))])
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["attempt_count"], 1)
        self.assertEqual(result["timing"]["generation_total_seconds"], 0.25)
        self.assertEqual(client.calls[0]["max_tokens"], 600)
        self.assertEqual(client.calls[0]["extra_body"]["thinking"]["type"], "disabled")
        self.assertEqual([e["event"] for e in events], ["attempt_started", "attempt_finished"])
        self.assertEqual(result["attempts"][0]["response_model"], "mock-server-model")

    def test_json_and_schema_retry_same_policy_and_usage(self):
        result, client, events = self.run_request([
            response("not json"), response('{}'), response(json.dumps(VALID))])
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["attempt_count"], 3)
        self.assertEqual(result["timing"]["retry_backoff_seconds"], 6)
        self.assertEqual(result["timing"]["generation_total_seconds"], 6.75)
        self.assertEqual(result["usage_all_attempts"]["prompt_tokens_known_sum"], 30)
        self.assertTrue(result["usage_all_attempts"]["total_tokens_complete"])
        self.assertEqual(client.calls[0], client.calls[1])
        self.assertEqual(client.calls[1], client.calls[2])

    def test_transient_errors_retry_but_missing_usage_is_unknown(self):
        result, client, _ = self.run_request([ApiError(429), response(json.dumps(VALID))])
        self.assertEqual(len(client.calls), 2)
        self.assertFalse(result["usage_all_attempts"]["total_tokens_complete"])
        self.assertEqual(result["usage_all_attempts"]["total_tokens_known_sum"], 15)

    def test_auth_permissions_and_bad_config_halt_without_retry(self):
        for code in (400, 401, 403, 404, 422):
            with self.subTest(code=code):
                result, client, events = self.run_request([ApiError(code)])
                self.assertTrue(result["halt_batch"])
                self.assertEqual(len(client.calls), 1)
                self.assertNotIn("SECRET_MUST_NOT_BE_LOGGED", json.dumps(events))

    def test_exhaustion_does_not_invent_an_abstention(self):
        result, client, _ = self.run_request([TimeoutError(), TimeoutError(), TimeoutError()])
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["model_response"])
        self.assertEqual(len(client.calls), 3)
        self.assertIsNone(result["usage_all_attempts"]["total_tokens_known_sum"])

    def test_truncation_not_accepted_or_retried_with_same_cap(self):
        result, client, _ = self.run_request([response(json.dumps(VALID), finish="length")])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(client.calls), 1)

    def test_semantic_wrongness_does_not_trigger_extra_generation(self):
        answer = {**VALID, "answer": "wrong answer", "citations": ["E999"]}
        result, client, _ = self.run_request([response(json.dumps(answer))])
        self.assertEqual(result["model_response"], answer)
        self.assertEqual(len(client.calls), 1)

    def test_log_failure_stops_before_network(self):
        clock = Clock()
        client = MockClient([], clock)
        def fail_log(event):
            raise OSError("disk full")
        with self.assertRaises(OSError):
            runtime.execute_question(client, {}, validate, fail_log, clock=clock, sleep=clock.sleep)
        self.assertEqual(client.calls, [])

    def test_gold_never_enters_messages(self):
        sample = {"company": "Company", "question": "Question?", "gold_answer": "GOLD_SENTINEL",
                  "justification": "JUSTIFICATION_SENTINEL", "evidence": ["GOLD_PAGE_SENTINEL"]}
        evidence = [{"citation_id": "E1", "doc_name": "D", "page_num": 1, "text": "real retrieved text",
                     "is_gold_page": "GOLD_FLAG_SENTINEL"}]
        for value in [None, evidence]:
            encoded = json.dumps(runner.make_messages(sample, "system", value))
            for secret in ("GOLD_SENTINEL", "JUSTIFICATION_SENTINEL", "GOLD_PAGE_SENTINEL", "GOLD_FLAG_SENTINEL"):
                self.assertNotIn(secret, encoded)
        self.assertIn("real retrieved text", encoded)

    def test_request_snapshot_not_mutated(self):
        messages = [{"role": "user", "content": "original"}]
        request = runtime.request_payload("mock", messages)
        digest = runtime.object_hash(request)
        messages[0]["content"] = "changed"
        self.assertEqual(runtime.object_hash(request), digest)
        request["extra_body"]["thinking"]["type"] = "changed"
        self.assertEqual(runtime.GENERATION["extra_body"]["thinking"]["type"], "disabled")

    def test_new_json_never_overwrites(self):
        with tempfile.TemporaryDirectory(prefix="qa_audit_test_") as directory:
            path = Path(directory)/"manifest.json"
            runner.write_new_json(path, {"original": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.write_new_json(path, {"original": False})
            self.assertEqual(path.read_bytes(), before)

    def test_paid_execution_requires_explicit_mode_and_limit(self):
        for argv in (["runner", "--execute"], ["runner", "--execute", "--mode", "llm_only"]):
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    runner.parse_args()
                self.assertEqual(error.exception.code, 2)

    def test_check_only_never_creates_client_or_results(self):
        # Synthetic fixtures, not a claim that the user's real corpus ran here.
        with tempfile.TemporaryDirectory(prefix="qa_audit_check_test_") as directory:
            root = Path(directory)
            source = root/"protocol_fixture.py"
            source.write_text("# synthetic protocol fixture\n", encoding="utf-8")
            dev = root/"data/processed/dev_30.jsonl"
            dev.parent.mkdir(parents=True)
            dev.write_text('\n'.join(json.dumps({"sample_id": f"financebench_id_{i:05d}",
                            "company": "mock", "question": "mock?"}) for i in range(30)), encoding="utf-8")
            protocol = types.ModuleType("experiment_protocol")
            protocol.__file__ = str(source)
            protocol.PROMPT_VERSION = "qa_protocol_v1"
            protocol.build_system_prompt = lambda mode: "mock system " + mode
            api = types.ModuleType("openai")
            def forbidden_client(**kwargs):
                raise AssertionError("API client must not be created in check-only")
            api.OpenAI = forbidden_client
            with patch.dict(sys.modules, {"experiment_protocol": protocol, "openai": api}), \
                 patch.object(runner, "load_environment", return_value=("deepseek-v4-flash", "https://example.invalid")), \
                 patch.object(sys, "argv", ["runner", "--check-only", "--mode", "llm_only", "--project-root", str(root)]), \
                 redirect_stdout(io.StringIO()) as output:
                self.assertEqual(runner.main(), 0)
            self.assertIn("CHECK_ONLY", output.getvalue())
            self.assertFalse((root/"results").exists())

    def test_both_runners_write_new_audited_logs_using_mock_transport(self):
        import numpy as np
        import hashlib
        for mode in ('llm_only', 'dense_rag'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix='qa_audit_flow_test_') as directory:
                root = Path(directory)
                data = root/'data/processed'
                data.mkdir(parents=True)
                dev = data/'dev_30.jsonl'
                dev.write_text('\n'.join(json.dumps({'sample_id':f'financebench_id_{i:05d}',
                    'company':'mock','question':'Question?', 'gold_answer':'GOLD_SENTINEL',
                    'justification':'JUSTIFICATION_SENTINEL'}) for i in range(30)), encoding='utf-8')
                corpus = [{'chunk_id':f'chunk_{i}', 'company':'mock', 'doc_name':'D',
                           'page_num':i+1, 'text':f'Evidence {i}'} for i in range(5)]
                corpus_path = data/'evidence_chunk_corpus.jsonl'
                corpus_path.write_text('\n'.join(json.dumps(row) for row in corpus),encoding='utf-8')
                np.save(data/'evidence_chunk_embeddings.npy', np.array([[1.,0.],[0.,1.],[.6,.8],[-1.,0.],[0.,-1.]]))
                metadata = {'model_name':'mock-embedding', 'embedding_dimension':2,
                    'chunk_ids_sha256':hashlib.sha256('\n'.join(r['chunk_id'] for r in corpus).encode()).hexdigest()}
                (data/'evidence_chunk_embeddings_meta.json').write_text(json.dumps(metadata),encoding='utf-8')
                protocol_path = root/'protocol_fixture.py'
                protocol_path.write_text('# synthetic fixture\n', encoding='utf-8')
                protocol = types.ModuleType('experiment_protocol')
                protocol.__file__ = str(protocol_path)
                protocol.PROMPT_VERSION = 'qa_protocol_v1'
                protocol.COMMON_SYSTEM_PROMPT = 'common mock prompt'
                protocol.build_system_prompt = lambda condition: 'common mock prompt\n' + condition
                protocol.QAAnswer = types.SimpleNamespace(model_validate=lambda value:
                    types.SimpleNamespace(model_dump=lambda:validate(value)))
                captured_clients = []
                api = types.ModuleType('openai')
                def make_client(**kwargs):
                    self.assertEqual(kwargs['max_retries'], 0)
                    self.assertEqual(kwargs['timeout'], 60)
                    answer = {**VALID, 'citations':[] if mode=='llm_only' else ['E1']}
                    client = MockClient([response(json.dumps(answer))],Clock())
                    captured_clients.append(client)
                    return client
                api.OpenAI = make_client
                embeddings = types.ModuleType('sentence_transformers')
                class FakeEmbedder:
                    device = 'cpu'
                    def __init__(self, name, *, local_files_only):
                        if not local_files_only:
                            raise AssertionError('Must not download embeddings')
                    def encode(self, questions, **kwargs):
                        return np.array([[1.,0.] for _ in questions])
                embeddings.SentenceTransformer = FakeEmbedder
                old = root/'results/raw_outputs/dense_rag_dev.jsonl'
                old.parent.mkdir(parents=True)
                old.write_text('LEGACY_SENTINEL\n', encoding='utf-8')
                with patch.dict(sys.modules, {'experiment_protocol':protocol,'openai':api,
                                               'sentence_transformers':embeddings}), \
                     patch.dict(runner.os.environ, {'LLM_API_KEY':'SECRET_SENTINEL'}), \
                     patch.object(runner,'load_environment',return_value=('deepseek-v4-flash','https://example.invalid')), \
                     patch('socket.socket.connect',side_effect=AssertionError('Network forbidden in tests')), \
                     patch.object(sys,'argv',['runner','--check-only','--project-root',str(root)]), \
                     redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(),0)
                    self.assertEqual(captured_clients,[])
                    self.assertFalse((root/'results/audited_runs').exists())
                    with patch.object(sys,'argv',['runner','--execute','--mode',mode,'--limit','1','--project-root',str(root)]):
                        self.assertEqual(runner.main(),0)
                self.assertEqual(len(captured_clients),1)
                self.assertTrue(captured_clients[0].closed)
                self.assertEqual(old.read_text(encoding='utf-8'),'LEGACY_SENTINEL\n')
                dirs = list((root/'results/audited_runs').iterdir())
                self.assertEqual(len(dirs),1)
                folder=dirs[0]
                self.assertEqual(len(list(folder.iterdir())),5)
                result = runner.read_jsonl(folder/'results.jsonl')[0]
                request = runner.read_jsonl(folder/'requests.jsonl')[0]
                manifest = json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
                summary = json.loads((folder/'run_summary.json').read_text(encoding='utf-8'))
                self.assertEqual(result['status'],'succeeded')
                self.assertEqual(result['protocol_violations'],[])
                self.assertEqual(request['request_sha256'],runtime.object_hash(request['request']))
                self.assertEqual(manifest['sources']['dev_data']['sha256'],runner.file_hash(dev))
                self.assertEqual(summary['succeeded'],1)
                self.assertTrue(summary['sources_unchanged_at_end'])
                self.assertNotIn('GOLD_SENTINEL',json.dumps(request))
                self.assertNotIn('JUSTIFICATION_SENTINEL',json.dumps(request))
                for path in folder.iterdir():
                    self.assertNotIn('SECRET_SENTINEL',path.read_text(encoding='utf-8'))
                if mode=='dense_rag':
                    self.assertEqual([r['rank'] for r in result['retrieved_results']],[1,2,3,4,5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
