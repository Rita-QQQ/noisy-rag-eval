"""Calculate final Dense-RAG metrics from the manually reviewed CSV.

Default project layout:
    src/calculate_dense_rag_metrics.py
    results/metrics/dense_rag_dev_manual_review.csv

Run from the project root:
    python src/calculate_dense_rag_metrics.py

You may also test another review file without replacing the canonical one:
    python src/calculate_dense_rag_metrics.py --input path/to/review.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


HIGH_CONFIDENCE_THRESHOLD = 0.8
EXPECTED_SAMPLE_COUNT = 30

REQUIRED_COLUMNS = {
    "sample_id",
    "question_type",
    "confidence",
    "abstain",
    "retrieval_hit_at_5",
    "cited_gold_page",
    "dataset_issue",
    "exclude_from_clean_subset_metric",
    "manual_correct",
    "citation_support_correct",
    "source_hallucination",
    "error_type",
    "failure_stage",
}

VALID_ERROR_TYPES = {
    "correct",
    "wrong_numeric",
    "wrong_conclusion",
    "incomplete",
    "abstention",
}

VALID_FAILURE_STAGES = {
    "none",
    "retrieval",
    "generation",
    "retrieval_and_generation",
}


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent if script_path.parent.name == "src" else script_path.parent
    default_input = project_root / "results" / "metrics" / "dense_rag_dev_manual_review.csv"
    default_output = project_root / "results" / "metrics"

    parser = argparse.ArgumentParser(
        description="Validate the Dense-RAG manual review table and calculate final metrics."
    )
    parser.add_argument("--input", type=Path, default=default_input, help="Manual review CSV")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Metric output directory")
    return parser.parse_args()


def normalize_bool(series: pd.Series, column: str) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    normalized = series.astype(str).str.strip().str.lower()
    invalid = sorted(set(normalized) - set(mapping))
    if invalid:
        raise ValueError(f"列 {column} 存在非法布尔值：{invalid}")
    return normalized.map(mapping).astype(bool)


def normalize_binary(series: pd.Series, column: str, allow_blank: bool = False) -> pd.Series:
    normalized = series.astype(str).str.strip()
    allowed = {"0", "1"} | ({""} if allow_blank else set())
    invalid = sorted(set(normalized) - allowed)
    if invalid:
        raise ValueError(f"列 {column} 只能填写 0/1{'/空白' if allow_blank else ''}，发现：{invalid}")
    return pd.to_numeric(normalized.replace("", pd.NA), errors="raise").astype("Int64")


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到人工审核表：{path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"人工审核表缺少列：{missing_columns}")
    if len(df) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"开发集应有 {EXPECTED_SAMPLE_COUNT} 行，实际为 {len(df)} 行")
    if df["sample_id"].str.strip().eq("").any():
        raise ValueError("sample_id 存在空值")
    if df["sample_id"].duplicated().any():
        duplicates = df.loc[df["sample_id"].duplicated(keep=False), "sample_id"].tolist()
        raise ValueError(f"sample_id 存在重复：{duplicates}")

    df = df.copy()
    for column in ("abstain", "retrieval_hit_at_5", "cited_gold_page", "exclude_from_clean_subset_metric"):
        df[column] = normalize_bool(df[column], column)
    df["manual_correct"] = normalize_binary(df["manual_correct"], "manual_correct")
    df["citation_support_correct"] = normalize_binary(
        df["citation_support_correct"], "citation_support_correct", allow_blank=True
    )
    df["source_hallucination"] = normalize_binary(df["source_hallucination"], "source_hallucination")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="raise")

    if not df["confidence"].between(0, 1, inclusive="both").all():
        raise ValueError("confidence 必须位于 0～1 之间")

    invalid_errors = sorted(set(df["error_type"]) - VALID_ERROR_TYPES)
    if invalid_errors:
        raise ValueError(f"error_type 存在非法值：{invalid_errors}")
    invalid_stages = sorted(set(df["failure_stage"]) - VALID_FAILURE_STAGES)
    if invalid_stages:
        raise ValueError(f"failure_stage 存在非法值：{invalid_stages}")

    problems: list[str] = []

    def add_problem(mask: pd.Series, message: str) -> None:
        ids = df.loc[mask, "sample_id"].tolist()
        if ids:
            problems.append(f"{message}：{ids}")

    correct = df["manual_correct"].eq(1)
    abstain = df["abstain"]
    cited_correct = df["citation_support_correct"]
    hallucinated = df["source_hallucination"].eq(1)

    add_problem(correct & ~df["error_type"].eq("correct"), "正确样本的 error_type 不是 correct")
    add_problem(~correct & df["error_type"].eq("correct"), "错误样本却标为 correct")
    add_problem(abstain & ~df["error_type"].eq("abstention"), "拒答样本的 error_type 不是 abstention")
    add_problem(~abstain & cited_correct.isna(), "已作答样本缺少 citation_support_correct")
    add_problem(abstain & cited_correct.notna(), "拒答样本的 citation_support_correct 应留空")
    add_problem(cited_correct.eq(1) & hallucinated, "同一样本不能同时是引用支持正确和来源幻觉")
    add_problem(
        df["failure_stage"].eq("none") & ~(correct & cited_correct.eq(1) & ~hallucinated),
        "failure_stage=none 与其他人工标签矛盾",
    )
    add_problem(
        ~df["retrieval_hit_at_5"] & df["failure_stage"].eq("generation"),
        "Top-5 未命中但 failure_stage 只标为 generation",
    )

    if problems:
        raise ValueError("人工审核表一致性检查失败：\n- " + "\n- ".join(problems))

    return df


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def summarize_subset(df: pd.DataFrame, name: str) -> dict[str, Any]:
    total = len(df)
    answered = df.loc[~df["abstain"]]
    wrong_answered = answered.loc[answered["manual_correct"].eq(0)]
    citation_applicable = df.loc[df["citation_support_correct"].notna()]

    correct_count = int(df["manual_correct"].sum())
    answered_correct_count = int(answered["manual_correct"].sum())
    citation_correct_count = int(citation_applicable["citation_support_correct"].sum())
    hallucination_count = int(df["source_hallucination"].sum())
    hallucination_when_answered = int(answered["source_hallucination"].sum())
    high_conf_wrong_count = int(
        wrong_answered["confidence"].ge(HIGH_CONFIDENCE_THRESHOLD).sum()
    )

    metrics = {
        "subset": name,
        "sample_count": total,
        "correct_count": correct_count,
        "accuracy": rate(correct_count, total),
        "answered_count": len(answered),
        "answered_correct_count": answered_correct_count,
        "answered_accuracy": rate(answered_correct_count, len(answered)),
        "abstain_count": int(df["abstain"].sum()),
        "abstain_rate": rate(int(df["abstain"].sum()), total),
        "retrieval_hit_at_5_count": int(df["retrieval_hit_at_5"].sum()),
        "retrieval_hit_at_5_rate": rate(int(df["retrieval_hit_at_5"].sum()), total),
        "cited_gold_page_count": int(df["cited_gold_page"].sum()),
        "cited_gold_page_rate": rate(int(df["cited_gold_page"].sum()), total),
        "citation_support_applicable_count": len(citation_applicable),
        "citation_support_correct_count": citation_correct_count,
        "citation_support_accuracy": rate(citation_correct_count, len(citation_applicable)),
        "source_hallucination_count": hallucination_count,
        "source_hallucination_rate_all": rate(hallucination_count, total),
        "source_hallucination_rate_when_answered": rate(
            hallucination_when_answered, len(answered)
        ),
        "mean_confidence": float(df["confidence"].mean()) if total else None,
        "mean_confidence_correct": (
            float(df.loc[df["manual_correct"].eq(1), "confidence"].mean())
            if correct_count
            else None
        ),
        "mean_confidence_wrong_answered": (
            float(wrong_answered["confidence"].mean()) if len(wrong_answered) else None
        ),
        "wrong_answered_count": len(wrong_answered),
        "high_confidence_wrong_count": high_conf_wrong_count,
        "high_confidence_wrong_rate": rate(high_conf_wrong_count, len(wrong_answered)),
    }
    return {key: rounded(value) for key, value in metrics.items()}


def build_by_type(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for question_type, group in df.groupby("question_type", sort=True):
        row = summarize_subset(group, question_type)
        row.pop("subset")
        rows.append({"question_type": question_type, **row})
    return pd.DataFrame(rows)


def count_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    result = df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="count")
    result["rate"] = result["count"] / len(df)
    return result


def print_summary(summary: dict[str, Any]) -> None:
    def pct(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2%}"

    print("=" * 78)
    print("Dense RAG 开发集正式评测结果")
    print("=" * 78)
    print(f"正确数量：{summary['correct_count']}/{summary['sample_count']}")
    print(f"回答准确率：{pct(summary['accuracy'])}")
    print(
        f"作答后准确率：{pct(summary['answered_accuracy'])} "
        f"({summary['answered_correct_count']}/{summary['answered_count']})"
    )
    print(f"拒答率：{pct(summary['abstain_rate'])}")
    print(f"检索 Hit@5：{pct(summary['retrieval_hit_at_5_rate'])}")
    print(
        f"引用支持准确率：{pct(summary['citation_support_accuracy'])} "
        f"({summary['citation_support_correct_count']}/"
        f"{summary['citation_support_applicable_count']})"
    )
    print(f"来源幻觉率（全部样本）：{pct(summary['source_hallucination_rate_all'])}")
    print(
        "来源幻觉率（已作答样本）："
        f"{pct(summary['source_hallucination_rate_when_answered'])}"
    )
    print(
        f"错误作答中的高置信错误率（置信度≥{HIGH_CONFIDENCE_THRESHOLD}）："
        f"{pct(summary['high_confidence_wrong_rate'])}"
    )


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    df = load_and_validate(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary = summarize_subset(df, "all_dev")
    clean_df = df.loc[~df["exclude_from_clean_subset_metric"]].copy()
    clean_summary = summarize_subset(clean_df, "clean_dev")
    by_type = build_by_type(df)
    error_types = count_table(df, "error_type")
    failure_stages = count_table(df, "failure_stage")

    report = {
        "system": "dense_rag",
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
        "input_file": str(input_path),
        "dataset_issue_count": int(df["dataset_issue"].str.strip().ne("").sum()),
        "excluded_from_clean_metric_count": int(df["exclude_from_clean_subset_metric"].sum()),
        "all_dev": all_summary,
        "clean_dev": clean_summary,
        "error_type_counts": dict(zip(error_types["error_type"], error_types["count"])),
        "failure_stage_counts": dict(zip(failure_stages["failure_stage"], failure_stages["count"])),
    }

    summary_path = output_dir / "dense_rag_dev_metrics.json"
    by_type_path = output_dir / "dense_rag_dev_by_type.csv"
    error_path = output_dir / "dense_rag_dev_error_types.csv"
    failure_path = output_dir / "dense_rag_dev_failure_stages.csv"

    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    by_type.to_csv(by_type_path, index=False, encoding="utf-8-sig")
    error_types.to_csv(error_path, index=False, encoding="utf-8-sig")
    failure_stages.to_csv(failure_path, index=False, encoding="utf-8-sig")

    print("人工审核表一致性检查：通过")
    print_summary(all_summary)
    print("\n清洁子集（排除有争议的标准答案）：")
    print(
        f"正确数量：{clean_summary['correct_count']}/{clean_summary['sample_count']}，"
        f"准确率：{clean_summary['accuracy']:.2%}"
    )
    print("\n按问题类型统计：")
    print(
        by_type[
            [
                "question_type",
                "sample_count",
                "accuracy",
                "answered_accuracy",
                "abstain_rate",
                "retrieval_hit_at_5_rate",
                "citation_support_accuracy",
                "source_hallucination_rate_all",
            ]
        ].to_string(index=False)
    )
    print("\n错误类型统计：")
    print(error_types.to_string(index=False))
    print("\n失败阶段统计：")
    print(failure_stages.to_string(index=False))
    print("\n结果已保存：")
    for path in (summary_path, by_type_path, error_path, failure_path):
        print(path)


if __name__ == "__main__":
    main()
