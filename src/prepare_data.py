from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# 固定随机种子，保证每次运行都得到相同的划分结果
RANDOM_SEED = 42


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "external"
    / "financebench"
    / "data"
    / "financebench_open_source.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

DEV_PATH = OUTPUT_DIR / "dev_30.jsonl"
TEST_PATH = OUTPUT_DIR / "test_120.jsonl"


# 读取原始数据
df = pd.read_json(RAW_DATA_PATH, lines=True)


# 在划分前做最基本的数据检查
assert len(df) == 150, "样本数量不是预期的150条"
assert df["financebench_id"].is_unique, "样本编号存在重复"
assert df["question"].notna().all(), "存在缺失问题"
assert df["answer"].notna().all(), "存在缺失答案"
assert df["evidence"].apply(len).ge(1).all(), "存在没有标准证据的问题"


processed_df = df.rename(
    columns={
        "financebench_id": "sample_id",
        "answer": "gold_answer",
        "question_reasoning": "reasoning",
    }
).copy()


# 增加证据数量字段，方便以后分析
processed_df["evidence_count"] = processed_df["evidence"].apply(len)


# 只保留本项目会使用的字段
selected_columns = [
    "sample_id",
    "company",
    "doc_name",
    "question_type",
    "reasoning",
    "question",
    "gold_answer",
    "justification",
    "evidence",
    "evidence_count",
]

processed_df = processed_df[selected_columns]


# 分层划分：
# dev取30条，剩余120条作为test
test_df, dev_df = train_test_split(
    processed_df,
    test_size=30,
    random_state=RANDOM_SEED,
    stratify=processed_df["question_type"],
)


# 按样本编号排序
dev_df = dev_df.sort_values("sample_id").reset_index(drop=True)
test_df = test_df.sort_values("sample_id").reset_index(drop=True)


# 确保输出文件夹存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 保存成JSONL
dev_df.to_json(
    DEV_PATH,
    orient="records",
    lines=True,
    force_ascii=False,
)

test_df.to_json(
    TEST_PATH,
    orient="records",
    lines=True,
    force_ascii=False,
)


# 检查两组数据是否有样本重叠
dev_ids = set(dev_df["sample_id"])
test_ids = set(test_df["sample_id"])

assert dev_ids.isdisjoint(test_ids), "开发集和测试集存在重叠"


print("=" * 60)
print("数据划分完成")
print("=" * 60)

print(f"开发集保存位置：{DEV_PATH}")
print(f"开发集样本数：{len(dev_df)}")
print(dev_df["question_type"].value_counts())

print("\n" + "-" * 60)

print(f"测试集保存位置：{TEST_PATH}")
print(f"测试集样本数：{len(test_df)}")
print(test_df["question_type"].value_counts())

print("\n开发集和测试集无重叠。")