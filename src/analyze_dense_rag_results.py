r"""分析Dense RAG开发集输出，并生成人工审核表。

运行位置：项目根目录
    python src\analyze_dense_rag_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEV_PATH = PROJECT_ROOT / "data" / "processed" / "dev_30.jsonl"
RESULT_PATH = (
    PROJECT_ROOT / "results" / "raw_outputs" / "dense_rag_dev.jsonl"
)
SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "dense_rag_dev_auto_summary.json"
)
BY_TYPE_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "dense_rag_dev_auto_by_type.csv"
)
REVIEW_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "dense_rag_dev_manual_review.csv"
)

MANUAL_COLUMNS = [
    "manual_correct",
    "citation_support_correct",
    "source_hallucination",
    "error_type",
    "failure_stage",
    "notes",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL解析失败：{path} 第{line_number}行") from exc
    return rows


def get_sample_id(sample: dict) -> str:
    sample_id = sample.get("sample_id") or sample.get("financebench_id")
    if not sample_id:
        raise KeyError("开发集样本缺少sample_id和financebench_id。")
    return str(sample_id)


def format_gold_evidence(sample: dict) -> str:
    sections = []
    for number, evidence in enumerate(sample.get("evidence", []), start=1):
        doc_name = evidence.get("doc_name") or sample.get("doc_name", "")
        page_num = evidence.get("evidence_page_num", "")
        evidence_text = evidence.get("evidence_text", "")
        sections.append(
            f"[Gold {number}]\n"
            f"Document: {doc_name}\n"
            f"Page: {page_num}\n"
            f"{evidence_text}"
        )
    return "\n\n".join(sections)


def format_retrieved_evidence(results: list[dict], cited_only: bool) -> str:
    sections = []
    for result in results:
        if cited_only and not result.get("cited_by_model", False):
            continue
        sections.append(
            f"[{result.get('citation_id', '')}]\n"
            f"Rank: {result.get('rank', '')}\n"
            f"Document: {result.get('doc_name', '')}\n"
            f"Page: {result.get('page_num', '')}\n"
            f"Gold page: {result.get('is_gold_page', False)}\n"
            f"Chunk: {result.get('chunk_id', '')}\n"
            f"{result.get('text', '')}"
        )
    return "\n\n".join(sections)


def safe_rate(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def build_auto_summary(records: list[dict]) -> dict:
    abstain_values = [bool(row["model_response"]["abstain"]) for row in records]
    confidence_values = [
        float(row["model_response"]["confidence"]) for row in records
    ]
    answered_records = [
        row for row in records if not bool(row["model_response"]["abstain"])
    ]

    total_prompt_tokens = sum(
        (row.get("usage") or {}).get("prompt_tokens") or 0 for row in records
    )
    total_completion_tokens = sum(
        (row.get("usage") or {}).get("completion_tokens") or 0 for row in records
    )

    return {
        "system": records[0].get("system") if records else None,
        "retriever": records[0].get("retriever") if records else None,
        "top_k": records[0].get("top_k") if records else None,
        "model": records[0].get("model") if records else None,
        "prompt_version": records[0].get("prompt_version") if records else None,
        "total_samples": len(records),
        "answered_count": len(answered_records),
        "abstain_count": int(sum(abstain_values)),
        "abstain_rate": safe_rate(abstain_values),
        "mean_confidence": float(np.mean(confidence_values)),
        "mean_confidence_when_answered": (
            float(
                np.mean(
                    [
                        float(row["model_response"]["confidence"])
                        for row in answered_records
                    ]
                )
            )
            if answered_records
            else None
        ),
        "retrieval_hit_at_5_rate": safe_rate(
            bool(row.get("retrieval_hit_at_5")) for row in records
        ),
        "cited_gold_page_rate": safe_rate(
            bool(row.get("cited_gold_page")) for row in records
        ),
        "cited_gold_page_rate_when_answered": safe_rate(
            bool(row.get("cited_gold_page")) for row in answered_records
        ),
        "answer_without_citation_count": sum(
            not bool(row["model_response"]["abstain"])
            and not bool(row["model_response"].get("citations"))
            for row in records
        ),
        "invalid_citation_label_count": sum(
            not bool(row.get("citation_labels_valid")) for row in records
        ),
        "mean_latency_seconds": float(
            np.mean([float(row.get("latency_seconds", 0.0)) for row in records])
        ),
        "total_prompt_tokens": int(total_prompt_tokens),
        "total_completion_tokens": int(total_completion_tokens),
        "total_tokens": int(total_prompt_tokens + total_completion_tokens),
    }


def preserve_existing_manual_columns(review_df: pd.DataFrame) -> pd.DataFrame:
    """重复运行时保留已经填写的人工审核内容。"""
    if not REVIEW_PATH.exists():
        for column in MANUAL_COLUMNS:
            review_df[column] = ""
        return review_df

    old_df = pd.read_csv(REVIEW_PATH, encoding="utf-8-sig", dtype=str).fillna("")
    if "sample_id" not in old_df.columns:
        raise ValueError("旧人工审核表缺少sample_id，无法安全保留审核内容。")

    old_df = old_df.set_index("sample_id")
    for column in MANUAL_COLUMNS:
        old_values = old_df[column] if column in old_df.columns else pd.Series(dtype=str)
        review_df[column] = [
            old_values.get(sample_id, "") for sample_id in review_df["sample_id"]
        ]
    return review_df


def main() -> None:
    print("=" * 78)
    print("Dense RAG开发集：自动统计与人工审核表生成")
    print("=" * 78)

    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"找不到Dense RAG结果：{RESULT_PATH}")

    records = load_jsonl(RESULT_PATH)
    dev_records = load_jsonl(DEV_PATH)

    if not records:
        raise ValueError("Dense RAG结果文件为空。")

    result_ids = [str(row["sample_id"]) for row in records]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("Dense RAG结果中存在重复sample_id。")

    dev_map = {get_sample_id(sample): sample for sample in dev_records}
    missing_ids = [sample_id for sample_id in result_ids if sample_id not in dev_map]
    if missing_ids:
        raise ValueError(f"以下结果无法在开发集中找到：{missing_ids}")

    auto_summary = build_auto_summary(records)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(auto_summary, file, ensure_ascii=False, indent=2)

    by_type_rows = []
    for question_type in sorted({str(row.get("question_type")) for row in records}):
        subset = [
            row for row in records if str(row.get("question_type")) == question_type
        ]
        summary = build_auto_summary(subset)
        by_type_rows.append(
            {
                "question_type": question_type,
                "sample_count": len(subset),
                "abstain_rate": summary["abstain_rate"],
                "mean_confidence": summary["mean_confidence"],
                "retrieval_hit_at_5_rate": summary["retrieval_hit_at_5_rate"],
                "cited_gold_page_rate": summary["cited_gold_page_rate"],
                "cited_gold_page_rate_when_answered": summary[
                    "cited_gold_page_rate_when_answered"
                ],
                "answer_without_citation_count": summary[
                    "answer_without_citation_count"
                ],
                "mean_latency_seconds": summary["mean_latency_seconds"],
            }
        )
    pd.DataFrame(by_type_rows).to_csv(
        BY_TYPE_PATH, index=False, encoding="utf-8-sig"
    )

    review_rows = []
    for record in records:
        sample_id = str(record["sample_id"])
        sample = dev_map[sample_id]
        response = record["model_response"]
        retrieved_results = record.get("retrieved_results", [])

        review_rows.append(
            {
                "sample_id": sample_id,
                "question_type": record.get("question_type"),
                "company": record.get("company"),
                "question": record.get("question"),
                "gold_answer": record.get("gold_answer"),
                "justification": sample.get("justification", ""),
                "gold_evidence_text": format_gold_evidence(sample),
                "predicted_answer": response.get("answer"),
                "confidence": response.get("confidence"),
                "abstain": response.get("abstain"),
                "model_citations": ", ".join(response.get("citations", [])),
                "reason": response.get("reason"),
                "retrieval_hit_at_5": record.get("retrieval_hit_at_5"),
                "cited_gold_page": record.get("cited_gold_page"),
                "citation_labels_valid": record.get("citation_labels_valid"),
                "cited_evidence_text": format_retrieved_evidence(
                    retrieved_results, cited_only=True
                ),
                "all_retrieved_evidence": format_retrieved_evidence(
                    retrieved_results, cited_only=False
                ),
                "dataset_issue": record.get("dataset_issue"),
                "exclude_from_clean_subset_metric": record.get(
                    "exclude_from_clean_subset_metric", False
                ),
            }
        )

    review_df = pd.DataFrame(review_rows)
    review_df = preserve_existing_manual_columns(review_df)
    review_df.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")

    print(f"system: {auto_summary['system']}")
    print(f"model: {auto_summary['model']}")
    print(f"prompt_version: {auto_summary['prompt_version']}")
    print(f"total_samples: {auto_summary['total_samples']}")
    print(f"answered_count: {auto_summary['answered_count']}")
    print(f"abstain_count: {auto_summary['abstain_count']}")
    print(f"abstain_rate: {auto_summary['abstain_rate']:.4f}")
    print(f"mean_confidence: {auto_summary['mean_confidence']:.4f}")
    print(
        "retrieval_hit_at_5_rate: "
        f"{auto_summary['retrieval_hit_at_5_rate']:.4f}"
    )
    print(
        "cited_gold_page_rate: "
        f"{auto_summary['cited_gold_page_rate']:.4f}"
    )
    print(
        "cited_gold_page_rate_when_answered: "
        f"{auto_summary['cited_gold_page_rate_when_answered']:.4f}"
    )
    print(
        "answer_without_citation_count: "
        f"{auto_summary['answer_without_citation_count']}"
    )
    print(
        "invalid_citation_label_count: "
        f"{auto_summary['invalid_citation_label_count']}"
    )
    print(f"mean_latency_seconds: {auto_summary['mean_latency_seconds']:.4f}")
    print(f"total_tokens: {auto_summary['total_tokens']}")

    print("\n结果已保存：")
    print(f"自动汇总：{SUMMARY_PATH}")
    print(f"按类型统计：{BY_TYPE_PATH}")
    print(f"人工审核表：{REVIEW_PATH}")
    print("\n注意：自动统计不代表答案正确，仍需填写人工审核列。")


if __name__ == "__main__":
    main()
