import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw_outputs"
    / "llm_only_dev.jsonl"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

SUMMARY_PATH = (
    METRICS_DIR
    / "llm_only_dev_summary.json"
)

REVIEW_PATH = (
    METRICS_DIR
    / "llm_only_manual_review.csv"
)


# 读取模型原始输出
df = pd.read_json(INPUT_PATH, lines=True)


# prediction原本是嵌套字典：
# {
#   "answer": "...",
#   "confidence": 0.7,
#   "abstain": false,
#   "reason": "..."
# }
#
# json_normalize将这些字段展开成普通列
prediction_df = pd.json_normalize(df["prediction"])


# 同样展开Token统计
usage_df = pd.json_normalize(df["usage"])


# 给Token字段加前缀，避免字段名混淆
usage_df = usage_df.add_prefix("usage_")


# 删除原来的嵌套列，再把展开后的列横向拼接回来
flat_df = pd.concat(
    [
        df.drop(columns=["prediction", "usage"]),
        prediction_df,
        usage_df,
    ],
    axis=1,
)


# 自动计算不涉及答案语义判断的指标
total_samples = len(flat_df)
abstain_count = int(flat_df["abstain"].sum())
answered_count = total_samples - abstain_count

summary = {
    "system": "llm_only",
    "model": flat_df["model"].iloc[0],
    "total_samples": total_samples,
    "answered_count": answered_count,
    "abstain_count": abstain_count,
    "abstain_rate": round(
        abstain_count / total_samples,
        4,
    ),
    "mean_confidence": round(
        float(flat_df["confidence"].mean()),
        4,
    ),
    "mean_confidence_when_answered": (
        round(
            float(
                flat_df.loc[
                    ~flat_df["abstain"],
                    "confidence",
                ].mean()
            ),
            4,
        )
        if answered_count > 0
        else None
    ),
    "mean_latency_seconds": round(
        float(flat_df["latency_seconds"].mean()),
        4,
    ),
    "total_prompt_tokens": int(
        flat_df["usage_prompt_tokens"].sum()
    ),
    "total_completion_tokens": int(
        flat_df["usage_completion_tokens"].sum()
    ),
    "total_tokens": int(
        flat_df["usage_total_tokens"].sum()
    ),
}


# 创建输出目录
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# 保存自动统计结果
with SUMMARY_PATH.open("w", encoding="utf-8") as file:
    json.dump(
        summary,
        file,
        ensure_ascii=False,
        indent=2,
    )


# 创建人工审核表
review_df = flat_df[
    [
        "sample_id",
        "question_type",
        "company",
        "question",
        "gold_answer",
        "answer",
        "confidence",
        "abstain",
        "reason",
    ]
].copy()


# 重命名，避免分不清哪个是模型答案
review_df = review_df.rename(
    columns={
        "answer": "predicted_answer",
    }
)


# 添加需要人工填写的字段
review_df["manual_correct"] = ""
review_df["source_hallucination"] = ""
review_df["error_type"] = ""
review_df["notes"] = ""


# utf-8-sig可以减少Excel打开中文CSV时乱码的概率
review_df.to_csv(
    REVIEW_PATH,
    index=False,
    encoding="utf-8-sig",
)


print("=" * 70)
print("LLM Only基线自动统计")
print("=" * 70)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\n结果已保存：")
print(f"自动统计：{SUMMARY_PATH}")
print(f"人工审核表：{REVIEW_PATH}")