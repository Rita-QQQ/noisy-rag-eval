"""Offline tests: local frozen files + fake API responses; no paid requests.

Run from the project root: python src/test_noise_audited.py
Only temporary test directories are written. Existing data/results are read-only.
"""
from __future__ import annotations

import builtins
import contextlib
import copy
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_noise_audited as runner

ROOT = Path(__file__).resolve().parents[1]
ANSWER = {"answer": "Synthetic answer", "confidence": 0.6, "abstain": False,
          "citations": ["E1"], "reason": "Synthetic explanation"}


def response(answer=None, *, raw=None, usage=True, finish="stop"):
    return SimpleNamespace(
        id="test-response", model="deepseek-v4-flash", system_fingerprint="test-fingerprint",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15} if usage else None,
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(
            content=json.dumps(ANSWER if answer is None else answer) if raw is None else raw))])


class FakeClient:
    def __init__(self, values=None):
        self.values = list(values or [])
        self.calls = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.calls.append(copy.deepcopy(request))
        value = self.values.pop(0) if self.values else response()
        if isinstance(value, BaseException):
            raise value
        return value

    def close(self):
        self.closed = True


class NoiseRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")):
            cls.frozen = runner.load_frozen(ROOT, ROOT / runner.PLAN_RELATIVE)
        cls.source_times = {key: Path(item["path"]).stat().st_mtime_ns
                            for key, item in cls.frozen["sources"].items()}

    @classmethod
    def tearDownClass(cls):
        if runner.changed_sources(cls.frozen["sources"]):
            raise AssertionError("Tests modified a protected source file")
        for key, item in cls.frozen["sources"].items():
            if Path(item["path"]).stat().st_mtime_ns != cls.source_times[key]:
                raise AssertionError("Tests changed source timestamps")

    def setUp(self):
        self.network = patch.object(socket, "socket", side_effect=AssertionError("network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def selected_requests(self, count=2):
        selected = runner.select_cases(self.frozen, count)
        requests = [runner.build_request(row, self.frozen["system_prompt"], "deepseek-v4-flash")
                    for row in selected]
        return selected, requests

    def fake_run(self, client):
        selected, requests = self.selected_requests()
        with tempfile.TemporaryDirectory(prefix="noise_runner_test_") as folder:
            path = Path(folder)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = runner.run_cases(self.frozen, selected, requests, client, path,
                                           "offline-test", "deepseek-v4-flash")
            logs = {p.name: p.read_text(encoding="utf-8") for p in path.iterdir()}
        return summary, logs

    def test_frozen_counts_and_hashes(self):
        self.assertEqual(len(self.frozen["cases"]), 120)
        self.assertEqual(len(self.frozen["order"]), 120)
        self.assertEqual(len(self.frozen["samples"]), 30)
        self.assertFalse(runner.changed_sources(self.frozen["sources"]))
        self.assertEqual(self.frozen["stats"]["exact_reference_length_matches"], 76)

    def test_two_questions_are_eight_balanced_cases_in_frozen_order(self):
        selected, _ = self.selected_requests()
        self.assertEqual([r["case_id"] for r in selected], self.frozen["order"][:8])
        self.assertEqual(list(dict.fromkeys(r["sample_id"] for r in selected)),
                         ["financebench_id_05915", "financebench_id_00563"])
        self.assertEqual(Counter(self.frozen["audits"][r["case_id"]]["replacement_count"]
                                 for r in selected), {0: 2, 1: 2, 2: 2, 3: 2})

    def test_full_selection_is_120_unique_cases(self):
        selected = runner.select_cases(self.frozen, 30)
        self.assertEqual(len({r["case_id"] for r in selected}), 120)
        self.assertEqual([r["case_id"] for r in selected], self.frozen["order"])
        for bad in (0, 31, -1, 2.5, True):
            with self.assertRaises(ValueError):
                runner.select_cases(self.frozen, bad)

    def test_all_zero_percent_requests_match_original_context_and_renderer(self):
        raw = runner.builder.read_jsonl(ROOT / runner.builder.SOURCE_FILES["rag"])
        for original in raw:
            case = self.frozen["cases"][runner.builder.case_id(original["sample_id"], 0)]
            request = runner.build_request(case, self.frozen["system_prompt"], "deepseek-v4-flash")
            expected = runner.shared.make_messages(original, self.frozen["system_prompt"], original["retrieved_results"])
            self.assertEqual(request["messages"], expected)

    def test_request_boundary_and_shared_parameters_for_all_120(self):
        for case in self.frozen["cases"].values():
            request = runner.build_request(case, self.frozen["system_prompt"], "deepseek-v4-flash")
            mi = case["model_input"]
            expected = runner.shared.make_messages(mi, self.frozen["system_prompt"], mi["evidence"])
            self.assertEqual(request, runner.runtime.request_payload("deepseek-v4-flash", expected))
            content = json.dumps(request)
            for forbidden in (case["case_id"], "noise_mask", "gold_answer", "replacement_fraction_of_blocks"):
                self.assertNotIn(forbidden, content)

    def test_outside_audit_metadata_does_not_change_request(self):
        case = copy.deepcopy(next(iter(self.frozen["cases"].values())))
        original = runner.build_request(case, self.frozen["system_prompt"], "deepseek-v4-flash")
        case.update(gold_answer="SECRET_GOLD_SENTINEL", noise_audit={"secret": "AUDIT_SENTINEL"})
        self.assertEqual(original, runner.build_request(case, self.frozen["system_prompt"], "deepseek-v4-flash"))

    def test_injected_input_field_and_text_tampering_rejected(self):
        case = copy.deepcopy(next(iter(self.frozen["cases"].values())))
        case["model_input"]["gold_answer"] = "never send"
        with self.assertRaises(ValueError):
            runner.build_request(case, self.frozen["system_prompt"], "deepseek-v4-flash")
        case = copy.deepcopy(next(iter(self.frozen["cases"].values())))
        case["model_input"]["evidence"][0]["text"] += "tampered"
        with self.assertRaises(ValueError):
            runner.build_request(case, self.frozen["system_prompt"], "deepseek-v4-flash")

    def test_tampered_plan_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="noise_plan_tamper_test_") as folder:
            target = Path(folder)
            for name in self.frozen["payloads"]:
                shutil.copyfile(ROOT / runner.PLAN_RELATIVE / name, target / name)
            changed = target / "noise_inputs.jsonl"
            changed.write_bytes(changed.read_bytes() + b"\n")
            before = changed.read_bytes()
            with self.assertRaisesRegex(ValueError, "不一致"):
                runner.load_frozen(ROOT, target)
            self.assertEqual(changed.read_bytes(), before)

    def test_changed_dependency_version_stops(self):
        hashes = {**runner.CODE_HASHES, "runtime": "0" * 64}
        with patch.object(runner, "CODE_HASHES", hashes), self.assertRaisesRegex(ValueError, "冻结版本"):
            runner.load_frozen(ROOT, ROOT / runner.PLAN_RELATIVE)

    def test_default_and_check_only_no_client_no_results(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"openai", "sentence_transformers", "transformers", "torch"}:
                raise AssertionError("Offline path imported a network/model client")
            return original_import(name, *args, **kwargs)

        for flags in ([], ["--check-only"], ["--check-only", "--limit-questions", "2"]):
            with tempfile.TemporaryDirectory(prefix="noise_offline_test_") as folder:
                with patch.object(runner, "load_frozen", return_value=self.frozen), \
                     patch.object(runner.shared, "load_environment", side_effect=AssertionError("Offline path read .env")), \
                     patch.object(builtins, "__import__", side_effect=guarded_import), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(["--project-root", folder] + flags), 0)
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_paid_execute_requires_explicit_limit(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for flags in (["--execute"], ["--execute", "--limit-questions", "0"],
                          ["--check-only", "--execute", "--limit-questions", "2"]):
                with self.assertRaises(SystemExit):
                    runner.parse_args(flags)

    def test_execute_entry_with_fake_sdk_freezes_plan_and_hides_key(self):
        client = FakeClient()
        constructor_calls = []

        def factory(**kwargs):
            constructor_calls.append(kwargs)
            return client

        with tempfile.TemporaryDirectory(prefix="noise_fake_execute_test_") as folder:
            root = Path(folder)
            legacy = root / "results/raw_outputs/sentinel.txt"
            legacy.parent.mkdir(parents=True)
            legacy.write_text("preserve old results", encoding="utf-8")
            with patch.object(runner, "load_frozen", return_value=self.frozen), \
                 patch.object(runner.shared, "load_environment", return_value=("deepseek-v4-flash", "https://example.invalid/v1")), \
                 patch.dict(os.environ, {"LLM_API_KEY": "DUMMY_NOT_A_REAL_KEY"}), \
                 patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=factory)}), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = runner.main(["--project-root", folder, "--execute", "--limit-questions", "2"])
            self.assertEqual(code, 0)
            self.assertTrue(client.closed)
            self.assertEqual(len(client.calls), 8)
            self.assertEqual(constructor_calls[0]["max_retries"], 0)
            self.assertEqual(constructor_calls[0]["timeout"], 60.0)
            runs = list((root / "results/noise_runs").iterdir())
            self.assertEqual(len(runs), 1)
            manifest = json.loads((runs[0] / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["selected_case_count"], 8)
            self.assertEqual(manifest["max_api_attempts"], 24)
            self.assertEqual(manifest["case_ids_in_execution_order"], self.frozen["order"][:8])
            for name, data in self.frozen["payloads"].items():
                self.assertEqual((runs[0] / "frozen_plan" / name).read_bytes(), data)
            for path in runs[0].rglob("*"):
                if path.is_file():
                    self.assertNotIn(b"DUMMY_NOT_A_REAL_KEY", path.read_bytes())
            self.assertEqual(legacy.read_text(encoding="utf-8"), "preserve old results")

    def test_mismatched_request_count_stops_before_api(self):
        selected, requests = self.selected_requests()
        client = FakeClient()
        with tempfile.TemporaryDirectory(prefix="noise_count_test_") as folder:
            with self.assertRaises(ValueError):
                runner.run_cases(self.frozen, selected, requests[:-1], client,
                                 Path(folder), "offline-test", "deepseek-v4-flash")
        self.assertEqual(client.calls, [])

    def test_eight_successes_journals_and_usage(self):
        client = FakeClient()
        summary, logs = self.fake_run(client)
        self.assertEqual(len(client.calls), 8)
        self.assertTrue(summary["complete"] and summary["all_succeeded"])
        self.assertEqual(summary["usage_all_attempts"]["total_tokens_known_sum"], 120)
        self.assertTrue(summary["usage_all_attempts"]["total_tokens_complete"])
        results = [json.loads(line) for line in logs["results.jsonl"].splitlines()]
        self.assertEqual(len(results), 8)
        self.assertTrue(all(row["noise_audit"]["case_id"] == row["case_id"] for row in results))
        self.assertTrue(all(row["protocol_violations"] == [] for row in results))

    def test_wrong_answer_or_bad_citation_never_retried(self):
        wrong = {**ANSWER, "answer": "Intentionally incorrect", "citations": ["E99"]}
        client = FakeClient([response(wrong)])
        summary, logs = self.fake_run(client)
        self.assertEqual(len(client.calls), 8)
        self.assertEqual(summary["succeeded"], 8)
        first = json.loads(logs["results.jsonl"].splitlines()[0])
        self.assertEqual(first["protocol_violations"], ["citation_label_not_allowed"])
        self.assertEqual(first["attempt_count"], 1)

    def test_json_retry_same_request_and_all_usage_counted(self):
        client = FakeClient([response(raw="{broken json"), response()])
        execute = runner.runtime.execute_question

        def no_sleep(*args, **kwargs):
            return execute(*args, **kwargs, sleep=lambda _: None)

        with patch.object(runner.runtime, "execute_question", side_effect=no_sleep):
            summary, _ = self.fake_run(client)
        self.assertEqual(len(client.calls), 9)
        self.assertEqual(client.calls[0], client.calls[1])
        self.assertEqual(summary["usage_all_attempts"]["total_tokens_known_sum"], 135)

    def test_auth_error_halts_unknown_usage_not_zero(self):
        error = RuntimeError("SECRET_DO_NOT_LOG")
        error.status_code = 401
        client = FakeClient([error])
        summary, logs = self.fake_run(client)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(len(summary["not_attempted_case_ids"]), 7)
        self.assertFalse(summary["complete"] or summary["all_succeeded"])
        self.assertIsNone(summary["usage_all_attempts"]["total_tokens_known_sum"])
        self.assertFalse(summary["usage_all_attempts"]["total_tokens_complete"])
        self.assertNotIn("SECRET_DO_NOT_LOG", "".join(logs.values()))

    def test_truncation_recorded_not_fabricated_as_abstention(self):
        client = FakeClient([response(finish="length")])
        summary, logs = self.fake_run(client)
        self.assertTrue(summary["complete"])
        self.assertFalse(summary["all_succeeded"])
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(len(client.calls), 8)
        first = json.loads(logs["results.jsonl"].splitlines()[0])
        self.assertIsNone(first["model_response"])

    def test_source_drift_stops_before_request(self):
        with patch.object(runner, "changed_sources", return_value=["dev"]):
            client = FakeClient()
            summary, _ = self.fake_run(client)
        self.assertEqual(len(client.calls), 0)
        self.assertFalse(summary["sources_unchanged_at_end"])
        self.assertEqual(len(summary["not_attempted_case_ids"]), 8)

    def test_summary_exclusive_write_never_overwrites(self):
        with tempfile.TemporaryDirectory(prefix="noise_write_test_") as folder:
            path = Path(folder) / "run_summary.json"
            runner.shared.write_new_json(path, {"old": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.shared.write_new_json(path, {"new": True})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
