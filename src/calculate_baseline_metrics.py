import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REVIEW_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "llm_only_manual_review.csv"
)

OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "llm_only_dev_metrics.json"
)

OUTPUT_BY_TYPE_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "llm_only_dev_by_type.csv"
)


df = pd.read_csv(REVIEW_PATH)


# ---------- 数据完整性检查 ----------

required_columns = [
    "sample_id",
    "question_type",
    "confidence",
    "abstain",
    "manual_correct",
    "source_hallucination",
    "error_type",
]

for column in required_columns:
    assert column in df.columns, f"缺少字段：{column}"

assert len(df) == 30, "审核表应当包含30条样本"

assert df["manual_correct"].notna().all(), (
    "manual_correct存在空值"
)

assert df["source_hallucination"].notna().all(), (
    "source_hallucination存在空值"
)

assert df["error_type"].notna().all(), (
    "error_type存在空值"
)

assert set(df["manual_correct"].unique()).issubset({0, 1}), (
    "manual_correct只能填写0或1"
)

assert set(
    df["source_hallucination"].unique()
).issubset({0, 1}), (
    "source_hallucination只能填写0或1"
)


# ---------- 总体指标 ----------

total_samples = len(df)
correct_count = int(df["manual_correct"].sum())
incorrect_count = total_samples - correct_count

answered_df = df[~df["abstain"]]
incorrect_answered_df = answered_df[
    answered_df["manual_correct"] == 0
]

high_confidence_errors = incorrect_answered_df[
    incorrect_answered_df["confidence"] >= 0.8
]


metrics = {
    "system": "llm_only",
    "total_samples": total_samples,

    "correct_count": correct_count,
    "incorrect_count": incorrect_count,
    "accuracy": round(
        correct_count / total_samples,
        4,
    ),

    "answered_count": int(len(answered_df)),
    "abstain_count": int(df["abstain"].sum()),
    "abstain_rate": round(
        float(df["abstain"].mean()),
        4,
    ),

    "answered_accuracy": round(
        float(answered_df["manual_correct"].mean()),
        4,
    ),

    "source_hallucination_count": int(
        df["source_hallucination"].sum()
    ),
    "source_hallucination_rate": round(
        float(df["source_hallucination"].mean()),
        4,
    ),

    "mean_confidence": round(
        float(df["confidence"].mean()),
        4,
    ),

    "mean_confidence_correct": round(
        float(
            df.loc[
                df["manual_correct"] == 1,
                "confidence",
            ].mean()
        ),
        4,
    ),

    "mean_confidence_incorrect": round(
        float(
            df.loc[
                df["manual_correct"] == 0,
                "confidence",
            ].mean()
        ),
        4,
    ),

    "high_confidence_error_count": int(
        len(high_confidence_errors)
    ),

    "high_confidence_error_rate_among_wrong_answers": (
        round(
            len(high_confidence_errors)
            / len(incorrect_answered_df),
            4,
        )
        if len(incorrect_answered_df) > 0
        else None
    ),

    "error_type_counts": {
        key: int(value)
        for key, value
        in df["error_type"].value_counts().items()
    },
}


# ---------- 按问题类型统计 ----------

by_type_df = (
    df.groupby("question_type")
    .agg(
        sample_count=("sample_id", "size"),
        accuracy=("manual_correct", "mean"),
        source_hallucination_rate=(
            "source_hallucination",
            "mean",
        ),
        abstain_rate=("abstain", "mean"),
        mean_confidence=("confidence", "mean"),
    )
    .reset_index()
)


# 将小数转成更整齐的四位
numeric_columns = [
    "accuracy",
    "source_hallucination_rate",
    "abstain_rate",
    "mean_confidence",
]

by_type_df[numeric_columns] = (
    by_type_df[numeric_columns].round(4)
)


# ---------- 保存结果 ----------

with OUTPUT_JSON_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metrics,
        file,
        ensure_ascii=False,
        indent=2,
    )


by_type_df.to_csv(
    OUTPUT_BY_TYPE_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ---------- 显示结果 ----------

print("=" * 70)
print("LLM Only开发集评测结果")
print("=" * 70)

print(f"正确数量：{correct_count}/{total_samples}")
print(f"回答准确率：{metrics['accuracy']:.2%}")
print(f"作答后准确率：{metrics['answered_accuracy']:.2%}")
print(f"拒答率：{metrics['abstain_rate']:.2%}")

print(
    "来源幻觉率："
    f"{metrics['source_hallucination_rate']:.2%}"
)

print(
    "正确答案平均置信度："
    f"{metrics['mean_confidence_correct']:.4f}"
)

print(
    "错误答案平均置信度："
    f"{metrics['mean_confidence_incorrect']:.4f}"
)

print(
    "错误作答中的高置信错误率："
    f"{metrics['high_confidence_error_rate_among_wrong_answers']:.2%}"
)

print("\n按问题类型统计：")
print(by_type_df.to_string(index=False))

print("\n错误类型统计：")
for error_type, count in metrics["error_type_counts"].items():
    print(f"{error_type}: {count}")

print("\n结果已保存：")
print(OUTPUT_JSON_PATH)
print(OUTPUT_BY_TYPE_PATH)