"""Run the frozen cross_company_replace_v1 plan; default is OFFLINE ONLY.

Paid requests require --execute AND --limit-questions. Each selected question
includes all four conditions in the precommitted order. No implicit resume,
new retrieval, new sampling, prompt edits, or writes to legacy results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import prepare_irrelevant_noise as builder
import qa_audit_runtime as runtime
import run_qa_audited as shared

VERSION = "frozen_noise_runner_v1"
PLAN_RELATIVE = "data/noise_experiments/cross_company_replace_v1"
CODE_HASHES = {
    "builder": "923a17f27bb3e583373cef02302b3221ac5059301e20d6d98b1b9a33e4f0f96f",
    "runtime": "765b3a8bafb5f7156a2e8657aa6492cac013bc2cb2047d60470fd7ce844287d2",
    "renderer": "0d673ecee981646bc181f04896193059b45486bc13c8cd05da85e8f8c919d79a",
    "protocol": "658b6840a252401ddb64bc70af5e273a9618471d55fc1e3174d6b83e86c58ce3",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def changed_sources(sources):
    changed = []
    for name, item in sources.items():
        try:
            same = shared.file_hash(Path(item["path"])) == item["sha256"]
        except OSError:
            same = False
        if not same:
            changed.append(name)
    return changed


def load_frozen(root, plan_dir):
    """Verify on disk, then use only the verified in-memory bytes/objects."""
    import experiment_protocol as protocol

    modules = {"builder": builder, "runtime": runtime, "renderer": shared, "protocol": protocol}
    paths = {key: Path(module.__file__).resolve() for key, module in modules.items()}
    paths.update({key: root / relative for key, relative in builder.SOURCE_FILES.items()})
    paths["noise_runner"] = Path(__file__).resolve()
    sources = {key: {"path": str(path), "sha256": shared.file_hash(path)} for key, path in paths.items()}
    for key, expected in {**CODE_HASHES, **builder.EXPECTED_SOURCE_HASHES}.items():
        require(sources[key]["sha256"] == expected,
                f"{paths[key].name}不同于冻结版本；停止。不要改哈希绕过检查，请先核对文件版本。")
    require(protocol.PROMPT_VERSION == "qa_protocol_v1", "提示词版本不同")
    dev, corpus, raw = (builder.read_jsonl(paths[key]) for key in ("dev", "corpus", "rag"))
    hashes = {key: sources[key]["sha256"] for key in builder.SOURCE_FILES}
    payloads, stats = builder.prepare_payloads(dev, corpus, raw, hashes)
    builder.verify_existing(plan_dir, payloads)
    for name, data in payloads.items():
        sources["plan/" + name] = {"path": str(plan_dir / name),
                                    "sha256": hashlib.sha256(data).hexdigest()}
    require(not changed_sources(sources), "校验期间文件变化；停止，不调用API")

    def jsonl(name):
        return [json.loads(line) for line in payloads[name].decode("utf-8").splitlines() if line.strip()]

    return {
        "sources": sources, "payloads": payloads, "stats": stats,
        "samples": {row["sample_id"]: row for row in dev},
        "cases": {row["case_id"]: row for row in jsonl("noise_inputs.jsonl")},
        "audits": {row["case_id"]: row for row in jsonl("noise_audit.jsonl")},
        "order": json.loads(payloads["execution_order.json"]),
        "plan_manifest": json.loads(payloads["noise_manifest.json"]),
        "system_prompt": protocol.build_system_prompt("rag"),
        "common_prompt_sha256": runtime.object_hash(protocol.COMMON_SYSTEM_PROMPT),
    }


def select_cases(frozen, limit_questions):
    require(type(limit_questions) is int and 1 <= limit_questions <= 30,
            "--limit-questions必须为1~30的整数")
    sample_order = list(dict.fromkeys(frozen["cases"][cid]["sample_id"] for cid in frozen["order"]))
    selected_ids = set(sample_order[:limit_questions])
    selected = [frozen["cases"][cid] for cid in frozen["order"]
                if frozen["cases"][cid]["sample_id"] in selected_ids]
    require(len(selected_ids) == limit_questions and len(selected) == 4 * limit_questions,
            "问题选择不完整")
    for sid in selected_ids:
        levels = [frozen["audits"][row["case_id"]]["replacement_count"]
                  for row in selected if row["sample_id"] == sid]
        require(sorted(levels) == [0, 1, 2, 3], "每题必须且只能包含四档各一次")
    return selected


def build_request(case, system_prompt, model):
    mi = case["model_input"]
    require(set(mi) == builder.MODEL_INPUT_FIELDS, "模型输入包含审核字段或缺少字段")
    require(case["model_input_sha256"] == runtime.object_hash(mi), "模型输入哈希不符")
    require(len(mi["evidence"]) == 5, "每份输入必须恰好5块")
    for number, block in enumerate(mi["evidence"], 1):
        require(set(block) == builder.EVIDENCE_FIELDS and block["citation_id"] == f"E{number}",
                "证据字段或引用编号改变")
    # Explicit allow-list: never pass case_id, noise labels, gold, masks, or
    # audit dictionaries into the shared prompt renderer.
    question = {key: mi[key] for key in ("company", "question")}
    messages = shared.make_messages(question, system_prompt, mi["evidence"])
    return runtime.request_payload(model, messages)


def validate_answer(value):
    import experiment_protocol as protocol
    from pydantic import ValidationError
    try:
        return protocol.QAAnswer.model_validate(value).model_dump()
    except ValidationError as exc:
        raise runtime.SchemaError("QAAnswer validation failed") from exc


def mechanical_violations(answer):
    if answer is None:
        return []
    violations = []
    if any(label not in {"E1", "E2", "E3", "E4", "E5"} for label in answer["citations"]):
        violations.append("citation_label_not_allowed")
    if not answer["abstain"] and not answer["citations"]:
        violations.append("answered_without_citation")
    return violations


def run_cases(frozen, selected, requests, client, run_dir, run_id, model):
    """Injected client permits fully offline tests. Caller owns client.close()."""
    require(len(selected) == len(requests) and len(selected) > 0, "请求数与选择case数不一致")
    results, attempts = [], []
    halted_reason = None
    for index, (case, request) in enumerate(zip(selected, requests), 1):
        if changed_sources(frozen["sources"]):
            halted_reason = "source_changed_before_next_case"
            break
        started = time.perf_counter()
        cid, sid = case["case_id"], case["sample_id"]
        audit = frozen["audits"][cid]
        sample = frozen["samples"][sid]
        request_id = uuid.uuid4().hex
        identity = {"run_id": run_id, "case_id": cid, "sample_id": sid, "request_id": request_id}
        request_hash = runtime.object_hash(request)
        shared.append_event(run_dir / "requests.jsonl", {
            **identity, "recorded_at_utc": runtime.utc_now(),
            "request_sha256": request_hash, "request": request})

        def emit(event):
            shared.append_event(run_dir / "attempts.jsonl", {**identity, **event})

        outcome = runtime.execute_question(client, request, validate_answer, emit)
        outcome["timing"]["case_processing_seconds"] = time.perf_counter() - started
        row = {
            **identity, "runner_version": VERSION, "runtime_version": runtime.RUNTIME_VERSION,
            "mode": "frozen_dense_rag", "model": model, "prompt_version": "qa_protocol_v1",
            "request_sha256": request_hash, "model_input_sha256": case["model_input_sha256"],
            "question_type": sample["question_type"], "company": sample["company"],
            "question": sample["question"], "gold_answer": sample["gold_answer"],
            "justification": sample.get("justification"), "gold_evidence": sample.get("evidence"),
            "evidence_context": case["model_input"]["evidence"],
            "noise_audit": audit,
            "protocol_violations": mechanical_violations(outcome["model_response"]), **outcome,
        }
        shared.append_event(run_dir / "results.jsonl", row)
        results.append(row)
        attempts.extend(outcome["attempts"])
        print(f"[{outcome['status']} {index:03d}/{len(selected):03d}] {cid} attempts={outcome['attempt_count']}", flush=True)
        if outcome["halt_batch"]:
            halted_reason = "fatal_api_or_runtime_error"
            break
    changed = changed_sources(frozen["sources"])
    planned_ids = [row["case_id"] for row in selected]
    attempted_ids = {row["case_id"] for row in results}
    summary = {
        "run_id": run_id, "finished_at_utc": runtime.utc_now(),
        "selected_questions": len({row["sample_id"] for row in selected}),
        "planned_cases": len(selected), "attempted_cases": len(results),
        "succeeded": sum(row["status"] == "succeeded" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "not_attempted_case_ids": [cid for cid in planned_ids if cid not in attempted_ids],
        "halted_reason": halted_reason, "changed_source_keys": changed,
        "sources_unchanged_at_end": not changed,
        "complete": len(results) == len(selected),
        "all_succeeded": len(results) == len(selected) and all(row["status"] == "succeeded" for row in results),
        "usage_all_attempts": runtime.usage_summary(attempts),
        "by_level": [],
        "meaning": "succeeded means request/schema success only, not answer correctness or citation support",
    }
    for level in range(4):
        planned = sum(frozen["audits"][cid]["replacement_count"] == level for cid in planned_ids)
        level_rows = [row for row in results if row["noise_audit"]["replacement_count"] == level]
        summary["by_level"].append({
            "replacement_count": level, "block_fraction": level / 5, "planned": planned,
            "succeeded": sum(row["status"] == "succeeded" for row in level_rows),
            "failed": sum(row["status"] == "failed" for row in level_rows),
            "not_attempted": planned - len(level_rows)})
    shared.write_new_json(run_dir / "run_summary.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--plan-dir", type=Path, default=Path(PLAN_RELATIVE))
    args = parser.parse_args(argv)
    if args.limit_questions is not None and not 1 <= args.limit_questions <= 30:
        parser.error("--limit-questions必须为1~30")
    if args.execute and args.limit_questions is None:
        parser.error("付费执行必须显式指定--execute和--limit-questions；不会默认跑120份")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = args.project_root.resolve()
    plan_dir = (root / args.plan_dir).resolve()
    frozen = load_frozen(root, plan_dir)
    # Pure offline checks use the specified target model, without loading .env
    # or requiring python-dotenv. Real execution must verify configured values.
    model, endpoint = "deepseek-v4-flash", None
    if args.execute:
        model, endpoint = shared.load_environment(root)
        require(model == "deepseek-v4-flash", "模型不是既定deepseek-v4-flash；停止")
    # Validate rendering for ALL 120 cases even for a small selected run.
    all_requests = {cid: build_request(frozen["cases"][cid], frozen["system_prompt"], model)
                    for cid in frozen["order"]}
    selected = select_cases(frozen, args.limit_questions or 30)
    requests = [all_requests[row["case_id"]] for row in selected]
    print("固定方案与来源哈希检查通过：30题、120份输入，每份5块；没有重新检索或抽样。")
    print("120份请求渲染检查通过：不发送标准答案、噪声比例或审核标签。")
    print("共享设置：qa_protocol_v1；deepseek-v4-flash；temperature=0；max_tokens=600；thinking=disabled。")
    print("最多3次尝试；SDK内部重试=0；错误答案或引用错误不会触发重试。")
    print(f"本次选择：{len(selected)//4}题 × 4档 = {len(selected)}份输入；每档{len(selected)//4}份。")
    if not args.execute:
        print("CHECK_ONLY：未读取.env；未创建API客户端；未调用API；未加载向量模型；未生成实验结果。")
        print("以上为预定模型设置；实际执行时才核验本地模型名、接口地址和密钥是否已配置。")
        print("到这里停下，把输出发回。下一步才是2题×4档的8份小测试。")
        return 0

    require(bool(os.getenv("LLM_API_KEY")), "未配置LLM_API_KEY；未调用API，不要上传.env")
    require(not changed_sources(frozen["sources"]), "准备期间输入文件变化；停止，不调用API")
    from openai import OpenAI

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = root / "results/noise_runs" / (builder.VERSION + "_" + run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = run_dir / "frozen_plan"
    snapshot.mkdir()
    for name, data in frozen["payloads"].items():
        with (snapshot / name).open("xb") as stream:
            stream.write(data)
    manifest = {
        "run_id": run_id, "created_at_utc": runtime.utc_now(), "runner_version": VERSION,
        "runtime_version": runtime.RUNTIME_VERSION, "plan_version": builder.VERSION,
        "mode": "frozen_dense_rag", "prompt_version": "qa_protocol_v1",
        "requested_model": model, "base_url": endpoint, "generation": runtime.GENERATION,
        "retry_policy": runtime.POLICY,
        "generation_policy_sha256": runtime.object_hash({"generation": runtime.GENERATION, "retry": runtime.POLICY}),
        "system_prompt": frozen["system_prompt"], "system_prompt_sha256": runtime.object_hash(frozen["system_prompt"]),
        "common_prompt_sha256": frozen["common_prompt_sha256"],
        "sources": frozen["sources"], "dependencies": shared.dependencies(),
        "case_ids_in_execution_order": [row["case_id"] for row in selected],
        "sample_ids": list(dict.fromkeys(row["sample_id"] for row in selected)),
        "planned_request_hashes": {row["case_id"]: runtime.object_hash(request) for row, request in zip(selected, requests)},
        "selected_case_count": len(selected), "max_api_attempts": len(selected) * runtime.POLICY["max_attempts"],
        "condition_counts": dict(Counter(frozen["audits"][row["case_id"]]["replacement_count"] for row in selected)),
        "retrieval": "none during this run; supplied original Top-5 frozen before replacement",
        "resume_policy": "none; every --execute creates a separate run and incurs new requests",
        "timing_definitions": {
            "api_seconds": "one SDK request; internal SDK retries disabled",
            "generation_total_seconds": "all attempts, schema checks, event logging and actual retry waits",
            "case_processing_seconds": "request journaling and generation; excludes source hashing, initial plan verification, request preparation and final result write",
            "retrieval_seconds": "not measured; no retrieval performed in this frozen-context experiment"},
        "limitations": frozen["plan_manifest"]["limitations"],
        "baseline_policy": "new 0% calls are the control; do not reuse historical 40% accuracy",
        "semantic_review": "preview sampling only; not certification of all 90 replacement assignments",
    }
    shared.write_new_json(run_dir / "run_manifest.json", manifest)
    print(f"EXECUTE：通常{len(selected)}次请求，含重试最多{manifest['max_api_attempts']}次；会产生费用。", flush=True)
    print("新运行目录：" + str(run_dir), flush=True)
    client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=endpoint,
                    max_retries=runtime.POLICY["sdk_max_retries"], timeout=runtime.POLICY["timeout_seconds"])
    try:
        summary = run_cases(frozen, selected, requests, client, run_dir, run_id, model)
    finally:
        client.close()
    ok = summary["all_succeeded"] and summary["sources_unchanged_at_end"]
    print(f"结束：成功{summary['succeeded']}，失败{summary['failed']}，未尝试{len(summary['not_attempted_case_ids'])}。")
    print("未修改旧结果。成功仅代表请求和结构有效，仍需人工审核答案与引用。")
    print("没有run_summary.json表示运行中断；有该文件也需检查all_succeeded和sources_unchanged_at_end。")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ImportError, KeyError) as exc:
        print(f"停止：{type(exc).__name__}；没有覆盖旧结果。")
        if not type(exc).__module__.startswith("openai"):
            print(str(exc))
        raise SystemExit(2)
