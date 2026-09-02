r"""在 FinanceBench 开发集上比较 Dense、BM25、RRF 和 CrossEncoder 重排。

运行位置：项目根目录
    python src\evaluate_retrieval_methods.py

本脚本只评测检索，不调用 DeepSeek API。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEV_PATH = PROJECT_ROOT / "data" / "processed" / "dev_30.jsonl"
CORPUS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "evidence_chunk_corpus.jsonl"
)
EMBEDDINGS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "evidence_chunk_embeddings.npy"
)
ISSUES_PATH = PROJECT_ROOT / "data" / "annotations" / "dataset_issues.jsonl"

RAW_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "raw_outputs" / "retrieval_methods_dev.jsonl"
)
SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "retrieval_methods_dev_summary.csv"
)
BY_TYPE_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "retrieval_methods_dev_by_type.csv"
)

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

CANDIDATE_K = 20
RRF_K = 60
TOP_K_VALUES = (1, 3, 5, 10)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL解析失败：{path} 第{line_number}行"
                ) from exc
    return rows


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_page_num(value) -> str:
    """把 40、40.0、'40' 统一为同一个页面编号。"""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip().casefold()


def page_key(doc_name, page_num) -> tuple[str, str]:
    return str(doc_name).strip().casefold(), normalize_page_num(page_num)


def lexical_tokens(text: str) -> list[str]:
    """供BM25和精确证据匹配使用的简单英文/数字分词。"""
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text).casefold())


def contains_token_sequence(container_text: str, evidence_text: str) -> bool:
    """忽略大小写、空白和常见标点，检查证据Token序列是否完整出现。"""
    container = lexical_tokens(container_text)
    evidence = lexical_tokens(evidence_text)

    if not evidence or len(evidence) > len(container):
        return False

    first = evidence[0]
    max_start = len(container) - len(evidence)
    for start in range(max_start + 1):
        if container[start] == first and container[start : start + len(evidence)] == evidence:
            return True
    return False


def gold_pages(sample: dict) -> set[tuple[str, str]]:
    pages = set()
    for evidence in sample.get("evidence", []):
        doc_name = evidence.get("doc_name") or sample.get("doc_name")
        page_num = evidence.get("evidence_page_num")
        if doc_name is not None and page_num is not None:
            pages.add(page_key(doc_name, page_num))
    return pages


def gold_evidence_texts(sample: dict) -> list[str]:
    texts = []
    for evidence in sample.get("evidence", []):
        text = evidence.get("evidence_text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def exact_chunk_indices(sample: dict, corpus: list[dict]) -> set[int]:
    """找出完整包含任一标准局部证据的Chunk。"""
    evidence_texts = gold_evidence_texts(sample)
    if not evidence_texts:
        return set()

    sample_gold_pages = gold_pages(sample)
    matches = set()

    for index, chunk in enumerate(corpus):
        chunk_page = page_key(chunk.get("doc_name"), chunk.get("page_num"))
        if chunk_page not in sample_gold_pages:
            continue

        text = chunk.get("text", "")
        if any(contains_token_sequence(text, evidence) for evidence in evidence_texts):
            matches.add(index)

    return matches


def stable_descending_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores), kind="mergesort")


def positions_from_order(order: np.ndarray, corpus_size: int) -> np.ndarray:
    positions = np.empty(corpus_size, dtype=np.int32)
    positions[order] = np.arange(1, corpus_size + 1, dtype=np.int32)
    return positions


def first_matching_rank(order: Iterable[int], matching_indices: set[int]) -> int | None:
    if not matching_indices:
        return None
    for rank, index in enumerate(order, start=1):
        if int(index) in matching_indices:
            return rank
    return None


def first_gold_page_rank(
    order: Iterable[int], corpus_page_keys: list[tuple[str, str]], sample_gold_pages
) -> int | None:
    for rank, index in enumerate(order, start=1):
        if corpus_page_keys[int(index)] in sample_gold_pages:
            return rank
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def build_passage(chunk: dict) -> str:
    return (
        f"Company: {chunk.get('company', '')}\n"
        f"Document: {chunk.get('doc_name', '')}\n"
        f"Page: {chunk.get('page_num', '')}\n"
        f"Content:\n{chunk.get('text', '')}"
    )


def load_issue_map() -> dict[str, dict]:
    if not ISSUES_PATH.exists():
        print(f"提示：未找到数据问题标注文件，将按全部样本评测：{ISSUES_PATH}")
        return {}

    rows = load_jsonl(ISSUES_PATH)
    issue_map = {}
    for row in rows:
        question_id = row.get("financebench_id")
        if not question_id:
            raise ValueError("dataset_issues.jsonl中存在缺少financebench_id的记录。")
        if question_id in issue_map:
            raise ValueError(f"dataset_issues.jsonl中存在重复记录：{question_id}")
        issue_map[question_id] = row
    return issue_map


def summarize(records: list[dict], subset_name: str) -> list[dict]:
    output = []
    methods = ("dense", "bm25", "rrf", "reranker")

    for method in methods:
        row = {
            "evaluation_subset": subset_name,
            "method": method,
            "sample_count": len(records),
        }

        for k in TOP_K_VALUES:
            row[f"page_hit_at_{k}"] = float(
                np.mean(
                    [
                        record["methods"][method]["first_gold_page_rank"] is not None
                        and record["methods"][method]["first_gold_page_rank"] <= k
                        for record in records
                    ]
                )
            )

        row["page_mrr"] = float(
            np.mean(
                [
                    reciprocal_rank(record["methods"][method]["first_gold_page_rank"])
                    for record in records
                ]
            )
        )

        exact_records = [record for record in records if record["exact_evaluable"]]
        row["exact_evaluable_count"] = len(exact_records)

        for k in TOP_K_VALUES:
            row[f"exact_hit_at_{k}"] = (
                float(
                    np.mean(
                        [
                            record["methods"][method]["first_exact_chunk_rank"]
                            is not None
                            and record["methods"][method]["first_exact_chunk_rank"] <= k
                            for record in exact_records
                        ]
                    )
                )
                if exact_records
                else np.nan
            )

        row["exact_mrr"] = (
            float(
                np.mean(
                    [
                        reciprocal_rank(
                            record["methods"][method]["first_exact_chunk_rank"]
                        )
                        for record in exact_records
                    ]
                )
            )
            if exact_records
            else np.nan
        )

        if method == "reranker":
            row["candidate_exact_recall"] = (
                float(
                    np.mean(
                        [record["candidate_pool_contains_exact"] for record in exact_records]
                    )
                )
                if exact_records
                else np.nan
            )
        else:
            row["candidate_exact_recall"] = np.nan

        output.append(row)

    return output


def main() -> None:
    print("=" * 78)
    print("FinanceBench开发集：四种检索方法批量对比")
    print("=" * 78)

    samples = load_jsonl(DEV_PATH)
    corpus = load_jsonl(CORPUS_PATH)
    corpus_embeddings = np.load(EMBEDDINGS_PATH)
    issue_map = load_issue_map()

    if len(corpus) != corpus_embeddings.shape[0]:
        raise ValueError(
            "语料数量和向量行数不一致："
            f"corpus={len(corpus)}, embeddings={corpus_embeddings.shape[0]}"
        )
    if corpus_embeddings.ndim != 2:
        raise ValueError(f"向量数组应为二维，实际形状：{corpus_embeddings.shape}")

    print(f"开发集问题数量：{len(samples)}")
    print(f"文本块数量：{len(corpus)}")
    print(f"语料向量形状：{corpus_embeddings.shape}")
    print(f"数据问题标注数量：{len(issue_map)}")
    print(f"Dense候选数：{CANDIDATE_K}")
    print(f"BM25候选数：{CANDIDATE_K}")
    print(f"RRF常数：{RRF_K}")

    corpus_texts = [str(chunk.get("text", "")) for chunk in corpus]
    corpus_page_keys = [
        page_key(chunk.get("doc_name"), chunk.get("page_num")) for chunk in corpus
    ]

    print("\n正在为BM25分词……")
    tokenized_corpus = [lexical_tokens(text) for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    print(f"正在加载Dense模型：{DENSE_MODEL_NAME}")
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)
    questions = [str(sample["question"]) for sample in samples]
    question_embeddings = dense_model.encode(
        questions,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if question_embeddings.shape[1] != corpus_embeddings.shape[1]:
        raise ValueError(
            "问题向量和语料向量维度不一致："
            f"questions={question_embeddings.shape}, corpus={corpus_embeddings.shape}"
        )

    print(f"正在加载重排模型：{RERANKER_MODEL_NAME}")
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    records = []
    print("\n正在逐题评测……")

    for sample_index, sample in enumerate(tqdm(samples, desc="Retrieval evaluation")):
        # 兼容两种数据结构：
        # - FinanceBench原始字段：financebench_id
        # - prepare_data.py规范化后的字段：sample_id
        question_id = sample.get("financebench_id") or sample.get("sample_id")
        if not question_id:
            raise KeyError(
                "开发集样本缺少financebench_id和sample_id，"
                f"实际字段为：{list(sample.keys())}"
            )
        question = str(sample["question"])
        sample_gold_pages = gold_pages(sample)
        sample_exact_indices = exact_chunk_indices(sample, corpus)

        dense_scores = corpus_embeddings @ question_embeddings[sample_index]
        dense_order = stable_descending_order(dense_scores)

        bm25_scores = np.asarray(bm25.get_scores(lexical_tokens(question)))
        bm25_order = stable_descending_order(bm25_scores)

        dense_positions = positions_from_order(dense_order, len(corpus))
        bm25_positions = positions_from_order(bm25_order, len(corpus))
        rrf_scores = (
            1.0 / (RRF_K + dense_positions)
            + 1.0 / (RRF_K + bm25_positions)
        )
        rrf_order = stable_descending_order(rrf_scores)

        candidate_indices = []
        seen = set()
        for index in list(dense_order[:CANDIDATE_K]) + list(
            bm25_order[:CANDIDATE_K]
        ):
            index = int(index)
            if index not in seen:
                seen.add(index)
                candidate_indices.append(index)

        pairs = [(question, build_passage(corpus[index])) for index in candidate_indices]
        reranker_scores = np.asarray(
            reranker.predict(pairs, show_progress_bar=False)
        ).reshape(-1)
        reranker_local_order = stable_descending_order(reranker_scores)
        reranker_order = np.asarray(
            [candidate_indices[int(i)] for i in reranker_local_order], dtype=np.int32
        )

        issue = issue_map.get(question_id, {})
        excluded_from_clean = bool(
            issue.get("exclude_from_clean_subset_metric", False)
        )

        method_orders = {
            "dense": dense_order,
            "bm25": bm25_order,
            "rrf": rrf_order,
            "reranker": reranker_order,
        }

        method_results = {}
        for method_name, order in method_orders.items():
            method_results[method_name] = {
                "first_gold_page_rank": first_gold_page_rank(
                    order, corpus_page_keys, sample_gold_pages
                ),
                "first_exact_chunk_rank": first_matching_rank(
                    order, sample_exact_indices
                ),
                "top_10_chunk_ids": [
                    corpus[int(index)]["chunk_id"] for index in order[:10]
                ],
            }

        records.append(
            {
                "financebench_id": question_id,
                "company": sample.get("company"),
                "question_type": sample.get("question_type"),
                "question": question,
                "dataset_issue": issue.get("dataset_issue"),
                "exclude_from_clean_subset_metric": excluded_from_clean,
                "gold_page_count": len(sample_gold_pages),
                "exact_evaluable": bool(sample_exact_indices),
                "exact_gold_chunk_count": len(sample_exact_indices),
                "candidate_pool_size": len(candidate_indices),
                "candidate_pool_contains_exact": bool(
                    sample_exact_indices.intersection(candidate_indices)
                ),
                "methods": method_results,
            }
        )

    save_jsonl(RAW_OUTPUT_PATH, records)

    all_summary_rows = summarize(records, "all_dev")
    clean_records = [
        record
        for record in records
        if not record["exclude_from_clean_subset_metric"]
    ]
    if len(clean_records) != len(records):
        all_summary_rows.extend(summarize(clean_records, "clean_dev"))

    summary_df = pd.DataFrame(all_summary_rows)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    by_type_rows = []
    for question_type in sorted(
        {str(record.get("question_type")) for record in records}
    ):
        type_records = [
            record for record in records if str(record.get("question_type")) == question_type
        ]
        rows = summarize(type_records, f"question_type:{question_type}")
        for row in rows:
            row["question_type"] = question_type
        by_type_rows.extend(rows)

    by_type_df = pd.DataFrame(by_type_rows)
    by_type_df.to_csv(BY_TYPE_PATH, index=False, encoding="utf-8-sig")

    display_columns = [
        "evaluation_subset",
        "method",
        "sample_count",
        "page_hit_at_5",
        "page_hit_at_10",
        "page_mrr",
        "exact_evaluable_count",
        "exact_hit_at_5",
        "exact_hit_at_10",
        "exact_mrr",
        "candidate_exact_recall",
    ]

    print("\n" + "=" * 78)
    print("批量检索评测完成")
    print("=" * 78)
    print(summary_df[display_columns].to_string(index=False))

    exact_evaluable_count = sum(record["exact_evaluable"] for record in records)
    print("\n说明：")
    print(
        f"- 30题中有{exact_evaluable_count}题存在至少一个完整包含标准局部证据的Chunk。"
    )
    print("- Exact指标只在这些可评测样本上计算，不会误罚无法由单Chunk容纳的证据。")
    print("- candidate_exact_recall只适用于reranker，表示正确Chunk是否进入第一阶段候选池。")

    print("\n结果已保存：")
    print(RAW_OUTPUT_PATH)
    print(SUMMARY_PATH)
    print(BY_TYPE_PATH)


if __name__ == "__main__":
    main()
