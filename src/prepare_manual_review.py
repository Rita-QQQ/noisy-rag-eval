from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw_outputs"
    / "llm_only_dev.jsonl"
)

DEV_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dev_30.jsonl"
)

REVIEW_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "llm_only_manual_review.csv"
)


def combine_evidence(evidence_list: list[dict]) -> str:
    """
    将一道题的多条标准证据合并成一段方便查看的文本。
    """

    evidence_parts = []

    for index, evidence in enumerate(evidence_list, start=1):
        text = evidence.get("evidence_text", "")
        doc_name = evidence.get("doc_name", "")
        page_num = evidence.get("evidence_page_num")

        evidence_parts.append(
            f"[Evidence {index}]\n"
            f"Document: {doc_name}\n"
            f"Page index: {page_num}\n"
            f"{text}"
        )

    return "\n\n".join(evidence_parts)


# 读取模型结果
result_df = pd.read_json(RESULT_PATH, lines=True)

prediction_df = pd.json_normalize(
    result_df["prediction"]
)

flat_result_df = pd.concat(
    [
        result_df.drop(columns=["prediction", "usage"]),
        prediction_df,
    ],
    axis=1,
)

flat_result_df = flat_result_df.rename(
    columns={
        "answer": "predicted_answer",
    }
)


# 读取开发集中的人工解释和标准证据
dev_df = pd.read_json(DEV_DATA_PATH, lines=True)

dev_df["gold_evidence_text"] = dev_df["evidence"].apply(
    combine_evidence
)


# 根据sample_id合并模型结果与原始数据
review_df = flat_result_df.merge(
    dev_df[
        [
            "sample_id",
            "justification",
            "gold_evidence_text",
        ]
    ],
    on="sample_id",
    how="left",
    validate="one_to_one",
)


# 调整人工审核表中的列顺序
review_df = review_df[
    [
        "sample_id",
        "question_type",
        "company",
        "question",
        "gold_answer",
        "justification",
        "gold_evidence_text",
        "predicted_answer",
        "confidence",
        "abstain",
        "reason",
    ]
]


# 添加人工标注列
review_df["manual_correct"] = ""
review_df["source_hallucination"] = ""
review_df["error_type"] = ""
review_df["notes"] = ""


review_df.to_csv(
    REVIEW_PATH,
    index=False,
    encoding="utf-8-sig",
)

print(f"人工审核表已更新：{REVIEW_PATH}")
print(f"样本数量：{len(review_df)}")
print(f"列数：{len(review_df.columns)}")
print("\n请不要在填写标注后重新运行本程序，否则会覆盖人工结果。")