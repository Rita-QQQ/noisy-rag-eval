"""New opt-in runner; NEVER writes legacy raw-output paths.

Default / --check-only: local checks only, no OpenAI client or embedding model.
--execute additionally requires --mode and --limit. Do not run it until you
intend to incur API usage. This version intentionally has no implicit resume.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from qa_audit_runtime import (GENERATION, POLICY, RUNTIME_VERSION, SchemaError,
                              canonical_json, execute_question, object_hash,
                              request_payload, utc_now)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{number} is not an object")
            rows.append(row)
    return rows


def write_new_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def append_event(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_samples(path):
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        sid = row.get("sample_id") or row.get("financebench_id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("开发集sample_id缺失或重复")
        if row.get("sample_id") and row.get("financebench_id") and row['sample_id'] != row['financebench_id']:
            raise ValueError("开发集两种ID字段不一致")
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("company", "question")):
            raise ValueError(f"{sid} 缺少公司或问题")
        row["sample_id"] = sid
        seen.add(sid)
    if len(rows) != 30:
        raise ValueError(f"本入口限dev_30：应为30题，实际{len(rows)}题")
    return rows


def inspect_dense(root):
    import numpy as np
    paths = {"corpus": root / "data/processed/evidence_chunk_corpus.jsonl",
             "embeddings": root / "data/processed/evidence_chunk_embeddings.npy",
             "embedding_metadata": root / "data/processed/evidence_chunk_embeddings_meta.json"}
    corpus = read_jsonl(paths["corpus"])
    metadata = json.loads(paths["embedding_metadata"].read_text(encoding="utf-8-sig"))
    vectors = np.load(paths["embeddings"], mmap_mode="r", allow_pickle=False)
    ids = [r["chunk_id"] for r in corpus]
    if len(set(ids)) != len(ids) or len(ids) < 5:
        raise ValueError("语料chunk_id重复或不足5条")
    if vectors.ndim != 2 or vectors.shape != (len(corpus), metadata["embedding_dimension"]):
        raise ValueError("语料、向量形状与元数据不匹配")
    if not np.isfinite(vectors).all():
        raise ValueError("向量含NaN或Infinity")
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != metadata["chunk_ids_sha256"]:
        raise ValueError("语料ID顺序与向量元数据不匹配")
    for row in corpus:
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("语料包含空文本")
        if any(k not in row for k in ("doc_name", "page_num")):
            raise ValueError("语料缺少文档/页码")
    if not metadata.get("model_name"):
        raise ValueError("向量模型名称为空")
    return paths, corpus, vectors, metadata


def make_messages(sample, system_prompt, evidence=None):
    user = f"Company:\n{sample['company']}\n\nQuestion:\n{sample['question']}"
    if evidence is not None:
        blocks = [f"[{e['citation_id']}]\nDocument: {e['doc_name']}\nPage: {e['page_num']}\nContent:\n{e['text']}" for e in evidence]
        user += "\n\nRetrieved evidence:\n" + "\n\n".join(blocks)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def retrieve(query, vectors, corpus):
    import numpy as np
    scores = vectors @ query
    selected = np.argsort(-scores, kind="mergesort")[:5]
    return [{"citation_id": f"E{rank}", "rank": rank,
             "score": float(scores[int(index)]),
             **{k: corpus[int(index)].get(k) for k in ("chunk_id", "company", "doc_name", "page_num", "text")}}
            for rank, index in enumerate(selected, 1)]


def dependencies():
    result = {"python": sys.version.split()[0]}
    for name in ("openai", "pydantic", "numpy", "sentence-transformers", "transformers", "torch", "python-dotenv"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def load_environment(root):
    # Standard configured credentials only; no client or network on this path.
    from dotenv import load_dotenv
    load_dotenv(root / ".env")
    model, endpoint = os.getenv("LLM_MODEL"), os.getenv("LLM_BASE_URL")
    if not model or not endpoint:
        raise ValueError("未配置LLM_MODEL或LLM_BASE_URL；不要把.env或密钥发到聊天中")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("LLM_BASE_URL须为不含用户名、密码或查询参数的HTTP(S)接口地址")
    endpoint = urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
    return model, endpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=["llm_only", "dense_rag"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 30:
        parser.error("--limit应在1~30之间")
    if args.execute and (args.mode is None or args.limit is None):
        parser.error("实际调用必须显式给出--execute、--mode和--limit；默认不会调用API")
    return args


def main():
    args = parse_args()
    root = args.project_root.resolve()
    # This module only defines prompts/schema. Never import legacy llm_client,
    # whose module-level OpenAI client would otherwise be created on dry runs.
    import experiment_protocol as protocol
    from pydantic import ValidationError
    if protocol.PROMPT_VERSION != "qa_protocol_v1":
        raise ValueError("本入口预期qa_protocol_v1；如需改变提示词，请先另立实验版本")
    model, endpoint = load_environment(root)
    if model != "deepseek-v4-flash":
        raise ValueError("当前LLM_MODEL与既定deepseek-v4-flash不同；已停止，未调用API")
    dev_path = root / "data/processed/dev_30.jsonl"
    samples = load_samples(dev_path)
    sources = {"dev_data": dev_path, "protocol_code": Path(protocol.__file__).resolve(),
               "runner_code": Path(__file__).resolve(),
               "runtime_code": Path(__file__).with_name("qa_audit_runtime.py")}
    dense = None
    if args.mode in (None, "dense_rag"):
        dense_paths, *dense = inspect_dense(root)
        sources.update(dense_paths)
    fingerprints = {key: {"path": str(path), "sha256": file_hash(path)} for key, path in sources.items()}
    prompts = {mode: protocol.build_system_prompt(mode) for mode in ("llm_only", "rag")}
    print("离线检查通过：开发集30题；提示词版本qa_protocol_v1。")
    if dense is not None:
        print(f"Dense语料/向量检查通过：{len(dense[0])}个chunk。没有加载或下载向量模型。")
    print("共享设置：temperature=0，max_tokens=600，thinking=disabled，最多3次尝试。")
    print("SDK内部重试=0；失败响应的usage未知时不会按零成本处理。")
    if not args.execute:
        print("CHECK_ONLY：未创建API客户端、未发网络请求、未生成实验结果。")
        print("现有代码和旧实验结果未修改。到这里停下，把输出发回即可。")
        return 0

    if not os.getenv("LLM_API_KEY"):
        raise ValueError("未配置LLM_API_KEY；未调用API")
    from openai import OpenAI
    setup_start = time.perf_counter()
    embedder = None
    if args.mode == "dense_rag":
        from sentence_transformers import SentenceTransformer
        # For real runs only. Require the existing local model cache; never
        # silently download a different revision during an audited run.
        embedder = SentenceTransformer(dense[2]["model_name"], local_files_only=True)
    setup_seconds = time.perf_counter() - setup_start
    query_vectors, query_encoding_seconds = None, 0.0
    if embedder is not None:
        # Keep the previous dense runner's all-dev batch encoding, including
        # batch_size=32. Time this shared setup separately, not once per sample.
        encoding_start = time.perf_counter()
        query_vectors = embedder.encode([sample["question"] for sample in samples],
                                         batch_size=32, show_progress_bar=False,
                                         convert_to_numpy=True, normalize_embeddings=True)
        query_encoding_seconds = time.perf_counter() - encoding_start
    if any(file_hash(path) != fingerprints[key]["sha256"] for key, path in sources.items()):
        raise ValueError("准备期间输入文件变化；未调用API，请保存文件后重试")
    client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=endpoint,
                    max_retries=POLICY["sdk_max_retries"], timeout=POLICY["timeout_seconds"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "Z_" + uuid.uuid4().hex[:8]
    run_dir = root / "results/audited_runs" / f"{args.mode}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    selected = samples[:args.limit]
    condition = "llm_only" if args.mode == "llm_only" else "rag"
    manifest = {
        "run_id": run_id, "created_at_utc": utc_now(), "runtime_version": RUNTIME_VERSION,
        "mode": args.mode, "prompt_version": protocol.PROMPT_VERSION,
        "requested_model": model, "base_url": endpoint, "generation": GENERATION,
        "retry_policy": POLICY, "generation_policy_sha256": object_hash({"generation": GENERATION, "retry": POLICY}),
        "system_prompt": prompts[condition], "system_prompt_sha256": object_hash(prompts[condition]),
        "common_prompt_sha256": object_hash(protocol.COMMON_SYSTEM_PROMPT),
        "sample_ids": [r["sample_id"] for r in selected], "sources": fingerprints,
        "dependencies": dependencies(), "model_setup_seconds": setup_seconds,
        "batch_query_encoding_seconds": query_encoding_seconds,
        "query_encoding_sample_count": len(samples) if embedder is not None else 0,
        "retrieval": None if embedder is None else {
            "top_k": 5, "embedding_model": dense[2]["model_name"], "embedding_device": str(embedder.device),
            "query_encoding": "all dev questions, batch_size=32; normalized; cached local model",
            "query_model_revision": "not independently pinned; local cache used",
            "tie_break": "stable mergesort; corpus order"},
        "timing_definitions": {
            "retrieval_seconds": "dense scoring + selecting/formatting evidence; query encoding measured separately; zero for llm_only",
            "generation_total_seconds": "all attempts + schema validation + event logging + actual retry waits",
            "api_seconds": "one SDK request only; SDK retries disabled",
            "sample_total_seconds": "retrieval + request construction/journaling + generation; excludes model setup, batch query encoding and final result write",
        },
        "scope": "new prospective run only; no retrospective claims about legacy runs",
        "compliance_policy": "retry JSON/schema errors; do not retry merely because answer/citations are wrong",
    }
    write_new_json(run_dir / "run_manifest.json", manifest)
    counts = {"selected": len(selected), "succeeded": 0, "failed": 0, "halted": False}
    print("EXECUTE：将产生API调用。新运行目录：" + str(run_dir), flush=True)

    def validate(value):
        try:
            return protocol.QAAnswer.model_validate(value).model_dump()
        except ValidationError as exc:
            raise SchemaError("QAAnswer validation failed") from exc

    try:
        for sample_index, sample in enumerate(selected):
            sample_start = time.perf_counter()
            retrieval_start = time.perf_counter()
            evidence = None if embedder is None else retrieve(query_vectors[sample_index], dense[1], dense[0])
            retrieval_seconds = 0.0 if evidence is None else time.perf_counter() - retrieval_start
            messages = make_messages(sample, prompts[condition], evidence)
            request = request_payload(model, messages)
            request_id = uuid.uuid4().hex
            append_event(run_dir / "requests.jsonl", {"sample_id": sample["sample_id"], "request_id": request_id,
                         "recorded_at_utc": utc_now(), "request_sha256": object_hash(request), "request": request})

            def emit(event):
                append_event(run_dir / "attempts.jsonl", {"sample_id": sample["sample_id"], "request_id": request_id, **event})

            outcome = execute_question(client, request, validate, emit)
            outcome["timing"].update(retrieval_seconds=retrieval_seconds,
                                     sample_total_seconds=time.perf_counter() - sample_start)
            answer = outcome["model_response"]
            allowed = set() if evidence is None else {e["citation_id"] for e in evidence}
            violations = []
            if answer is not None:
                if any(c not in allowed for c in answer["citations"]):
                    violations.append("citation_label_not_allowed")
                if evidence is not None and not answer["abstain"] and not answer["citations"]:
                    violations.append("answered_without_citation")
            result = {"sample_id": sample["sample_id"], "run_id": run_id, "request_id": request_id,
                      "mode": args.mode, "model": model, "prompt_version": protocol.PROMPT_VERSION,
                      "runtime_version": RUNTIME_VERSION, "request_sha256": object_hash(request),
                      "question_type": sample.get("question_type"), "company": sample["company"],
                      "question": sample["question"], "gold_answer": sample.get("gold_answer", sample.get("answer")),
                      "justification": sample.get("justification"), "retrieved_results": evidence,
                      "protocol_violations": violations, **outcome}
            append_event(run_dir / "results.jsonl", result)
            counts[outcome["status"]] += 1
            print(f"[{outcome['status']}] {sample['sample_id']} attempts={outcome['attempt_count']}", flush=True)
            if outcome["halt_batch"]:
                counts["halted"] = True
                print("遇到鉴权、权限、请求配置或程序错误，已停止整批；请检查日志中的错误类别。")
                break
    finally:
        client.close()
    counts["sources_unchanged_at_end"] = all(file_hash(path) == fingerprints[key]["sha256"] for key, path in sources.items())
    write_new_json(run_dir / "run_summary.json", {"finished_at_utc": utc_now(), **counts})
    print("运行结果已保存；未修改旧输出。没有run_summary.json的目录视为中断运行。")
    return 0 if counts["succeeded"] == len(selected) and counts["sources_unchanged_at_end"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ImportError, KeyError) as exc:
        print(f"停止：{type(exc).__name__}。请检查本地路径、依赖、协议和配置；原始结果未修改。")
        # Avoid printing arbitrary provider exception text containing secrets.
        if not type(exc).__module__.startswith("openai"):
            print(str(exc))
        raise SystemExit(2)
