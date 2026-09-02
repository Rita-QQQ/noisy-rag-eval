r"""在30题开发集上运行 Dense Top-5 基础RAG。

运行位置：项目根目录
    python src\run_dense_rag_dev.py

特点：
- 使用和LLM Only相同的qa_protocol_v1回答协议；
- 标准答案和gold页面不进入提示词；
- 每成功完成一题立即写入JSONL；
- 再次运行时自动跳过已经成功完成的样本。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from experiment_protocol import PROMPT_VERSION, QAAnswer, build_system_prompt
from llm_client import MODEL_NAME as LLM_MODEL_NAME, call_llm_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEV_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dev_30.jsonl"
CORPUS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "evidence_chunk_corpus.jsonl"
)
EMBEDDINGS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "evidence_chunk_embeddings.npy"
)
EMBEDDINGS_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evidence_chunk_embeddings_meta.json"
)
ISSUES_PATH = PROJECT_ROOT / "data" / "annotations" / "dataset_issues.jsonl"

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "raw_outputs" / "dense_rag_dev.jsonl"
)
ERROR_PATH = (
    PROJECT_ROOT / "results" / "raw_outputs" / "dense_rag_dev_errors.jsonl"
)

SYSTEM_NAME = "dense_rag"
RETRIEVER_NAME = "dense"
TOP_K = 5
MAX_OUTPUT_TOKENS = 600
MAX_ATTEMPTS = 3


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL解析失败：{path} 第{line_number}行") from exc
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def calculate_chunk_id_hash(records: list[dict]) -> str:
    chunk_id_text = "\n".join(record["chunk_id"] for record in records)
    return hashlib.sha256(chunk_id_text.encode("utf-8")).hexdigest()


def normalize_page_num(value) -> int | str:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return str(value).strip()


def page_key(doc_name, page_num) -> tuple[str, int | str]:
    return str(doc_name).strip().casefold(), normalize_page_num(page_num)


def get_gold_pages(sample: dict) -> set[tuple[str, int | str]]:
    """仅用于API回答完成后的离线统计。"""
    pages = set()
    for evidence in sample.get("evidence", []):
        doc_name = evidence.get("doc_name") or sample.get("doc_name")
        page_num = evidence.get("evidence_page_num")
        if doc_name is not None and page_num is not None:
            pages.add(page_key(doc_name, page_num))
    return pages


def load_issue_map() -> dict[str, dict]:
    if not ISSUES_PATH.exists():
        return {}
    rows = load_jsonl(ISSUES_PATH)
    return {
        row["financebench_id"]: row
        for row in rows
        if row.get("financebench_id")
    }


def get_sample_id(sample: dict) -> str:
    sample_id = sample.get("financebench_id") or sample.get("sample_id")
    if not sample_id:
        raise KeyError(
            "样本缺少financebench_id和sample_id，"
            f"实际字段为：{list(sample.keys())}"
        )
    return str(sample_id)


def validate_existing_records(records: list[dict]) -> set[str]:
    completed_ids = set()
    for record in records:
        if record.get("system") != SYSTEM_NAME:
            raise ValueError(
                f"已有输出system不一致：{record.get('system')} != {SYSTEM_NAME}"
            )
        if record.get("retriever") != RETRIEVER_NAME:
            raise ValueError("已有输出retriever与本次运行不一致。")
        if record.get("top_k") != TOP_K:
            raise ValueError("已有输出Top-K与本次运行不一致。")
        if record.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(
                "已有输出Prompt版本与本次运行不一致。"
                "请不要把不同实验混在同一个JSONL文件中。"
            )
        if record.get("model") != LLM_MODEL_NAME:
            raise ValueError("已有输出模型名称与本次运行不一致。")

        sample_id = record.get("sample_id")
        if not sample_id:
            raise ValueError("已有输出存在缺少sample_id的记录。")
        if sample_id in completed_ids:
            raise ValueError(f"已有输出存在重复样本：{sample_id}")
        completed_ids.add(sample_id)
    return completed_ids


def call_with_retry(messages: list[dict[str, str]]):
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call_llm_json(messages, max_tokens=MAX_OUTPUT_TOKENS)
        except Exception as exc:  # API连接或响应格式异常
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                wait_seconds = 2 * attempt
                print(
                    f"请求失败，第{attempt}/{MAX_ATTEMPTS}次；"
                    f"{wait_seconds}秒后重试：{exc}"
                )
                time.sleep(wait_seconds)
    raise RuntimeError(f"连续{MAX_ATTEMPTS}次调用失败") from last_error


def main() -> None:
    print("=" * 78)
    print("Dense Top-5 基础RAG：30题开发集批量运行")
    print("=" * 78)

    dev_df = pd.read_json(DEV_DATA_PATH, lines=True)
    samples = dev_df.to_dict(orient="records")
    corpus = load_jsonl(CORPUS_PATH)
    corpus_embeddings = np.load(EMBEDDINGS_PATH)
    issue_map = load_issue_map()

    with EMBEDDINGS_METADATA_PATH.open("r", encoding="utf-8") as file:
        embeddings_metadata = json.load(file)

    if corpus_embeddings.ndim != 2:
        raise ValueError(f"语料向量形状异常：{corpus_embeddings.shape}")
    if corpus_embeddings.shape[0] != len(corpus):
        raise ValueError("语料数量与向量行数不一致。")
    if corpus_embeddings.shape[1] != embeddings_metadata["embedding_dimension"]:
        raise ValueError("语料向量维度与元数据不一致。")
    if calculate_chunk_id_hash(corpus) != embeddings_metadata["chunk_ids_sha256"]:
        raise ValueError("语料顺序与向量文件不一致，请重新生成向量。")

    existing_records = load_jsonl(OUTPUT_PATH)
    completed_ids = validate_existing_records(existing_records)

    print(f"开发集样本数：{len(samples)}")
    print(f"已经成功完成：{len(completed_ids)}")
    print(f"本次待运行：{len(samples) - len(completed_ids)}")
    print(f"检索方法：{RETRIEVER_NAME} Top-{TOP_K}")
    print(f"向量模型：{embeddings_metadata['model_name']}")
    print(f"LLM模型：{LLM_MODEL_NAME}")
    print(f"Prompt版本：{PROMPT_VERSION}")
    print(f"输出位置：{OUTPUT_PATH}")

    if len(completed_ids) == len(samples):
        print("全部样本已经完成，无需重复调用API。")
        return

    print("\n正在加载向量模型并批量生成问题向量……")
    embedding_model = SentenceTransformer(embeddings_metadata["model_name"])
    questions = [str(sample["question"]) for sample in samples]
    query_embeddings = embedding_model.encode(
        questions,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    system_prompt = build_system_prompt("rag")
    success_this_run = 0
    failure_this_run = 0

    for sample_index, sample in enumerate(tqdm(samples, desc="Dense RAG")):
        sample_id = get_sample_id(sample)
        if sample_id in completed_ids:
            continue

        question = str(sample["question"])
        similarity_scores = corpus_embeddings @ query_embeddings[sample_index]
        top_indices = np.argsort(-similarity_scores, kind="mergesort")[:TOP_K]

        retrieved_results = []
        evidence_sections = []
        for rank, corpus_index in enumerate(top_indices, start=1):
            corpus_index = int(corpus_index)
            chunk = corpus[corpus_index]
            citation_id = f"E{rank}"

            retrieved_results.append(
                {
                    "citation_id": citation_id,
                    "rank": rank,
                    "score": float(similarity_scores[corpus_index]),
                    "chunk_id": chunk["chunk_id"],
                    "company": chunk.get("company"),
                    "doc_name": chunk["doc_name"],
                    "page_num": normalize_page_num(chunk["page_num"]),
                    "text": chunk["text"],
                }
            )

            evidence_sections.append(
                f"[{citation_id}]\n"
                f"Document: {chunk['doc_name']}\n"
                f"Page: {chunk['page_num']}\n"
                f"Content:\n{chunk['text']}"
            )

        user_prompt = (
            f"Company:\n{sample['company']}\n\n"
            f"Question:\n{question}\n\n"
            "Retrieved evidence:\n"
            + "\n\n".join(evidence_sections)
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        start_time = time.perf_counter()
        try:
            raw_result, usage = call_with_retry(messages)
            validated_result = QAAnswer.model_validate(raw_result)
        except Exception as exc:
            latency = time.perf_counter() - start_time
            failure_this_run += 1
            append_jsonl(
                ERROR_PATH,
                {
                    "sample_id": sample_id,
                    "system": SYSTEM_NAME,
                    "model": LLM_MODEL_NAME,
                    "prompt_version": PROMPT_VERSION,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "latency_seconds": latency,
                },
            )
            print(f"\n样本失败但继续运行：{sample_id}：{exc}")
            continue

        latency = time.perf_counter() - start_time
        allowed_citations = {
            result["citation_id"] for result in retrieved_results
        }
        invalid_citations = [
            citation
            for citation in validated_result.citations
            if citation not in allowed_citations
        ]

        gold_pages = get_gold_pages(sample)
        retrieval_hit = any(
            page_key(result["doc_name"], result["page_num"]) in gold_pages
            for result in retrieved_results
        )

        cited_results = [
            result
            for result in retrieved_results
            if result["citation_id"] in validated_result.citations
        ]
        cited_gold_page = any(
            page_key(result["doc_name"], result["page_num"]) in gold_pages
            for result in cited_results
        )

        for result in retrieved_results:
            result["is_gold_page"] = (
                page_key(result["doc_name"], result["page_num"]) in gold_pages
            )
            result["cited_by_model"] = (
                result["citation_id"] in validated_result.citations
            )

        issue = issue_map.get(sample_id, {})
        output_record = {
            "sample_id": sample_id,
            "system": SYSTEM_NAME,
            "retriever": RETRIEVER_NAME,
            "top_k": TOP_K,
            "embedding_model": embeddings_metadata["model_name"],
            "model": LLM_MODEL_NAME,
            "prompt_version": PROMPT_VERSION,
            "question_type": sample.get("question_type"),
            "company": sample.get("company"),
            "doc_name": sample.get("doc_name"),
            "question": question,
            "gold_answer": sample.get("gold_answer", sample.get("answer")),
            "dataset_issue": issue.get("dataset_issue"),
            "exclude_from_clean_subset_metric": bool(
                issue.get("exclude_from_clean_subset_metric", False)
            ),
            "model_response": validated_result.model_dump(),
            "retrieval_hit_at_5": retrieval_hit,
            "cited_gold_page": cited_gold_page,
            "citation_labels_valid": not invalid_citations,
            "invalid_citations": invalid_citations,
            "retrieved_results": retrieved_results,
            "usage": usage,
            "latency_seconds": latency,
        }
        append_jsonl(OUTPUT_PATH, output_record)
        completed_ids.add(sample_id)
        success_this_run += 1

    total_records = load_jsonl(OUTPUT_PATH)
    total_prompt_tokens = sum(
        (record.get("usage") or {}).get("prompt_tokens") or 0
        for record in total_records
    )
    total_completion_tokens = sum(
        (record.get("usage") or {}).get("completion_tokens") or 0
        for record in total_records
    )

    print("\n" + "=" * 78)
    print("Dense RAG开发集运行结束")
    print("=" * 78)
    print(f"本次成功：{success_this_run}")
    print(f"本次失败：{failure_this_run}")
    print(f"累计成功：{len(total_records)} / {len(samples)}")
    print(f"累计输入Token：{total_prompt_tokens}")
    print(f"累计输出Token：{total_completion_tokens}")
    print(f"成功结果：{OUTPUT_PATH}")
    if failure_this_run:
        print(f"错误日志：{ERROR_PATH}")
        print("失败样本没有写入成功结果，重新运行时会自动重试。")


if __name__ == "__main__":
    main()
