from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "external"
    / "financebench"
    / "data"
    / "financebench_open_source.jsonl"
)


df = pd.read_json(DATA_PATH, lines=True)


print("=" * 60)
print("一、数据集基本信息")
print("=" * 60)

print(f"样本总数：{len(df)}")
print(f"公司数量：{df['company'].nunique()}")
print(f"文档数量：{df['doc_name'].nunique()}")
print(f"问题是否重复：{df['question'].duplicated().sum()}")


print("\n" + "=" * 60)
print("二、缺失值统计")
print("=" * 60)

missing_counts = df.isna().sum()
print(missing_counts)


print("\n" + "=" * 60)
print("三、问题类型分布")
print("=" * 60)

question_type_counts = df["question_type"].value_counts(dropna=False)
print(question_type_counts)


print("\n" + "=" * 60)
print("四、推理类型分布")
print("=" * 60)

reasoning_counts = df["question_reasoning"].value_counts(dropna=False)
print(reasoning_counts)


print("\n" + "=" * 60)
print("五、每道题的证据数量")
print("=" * 60)

# evidence字段中的每个元素都是一个列表

df["evidence_count"] = df["evidence"].apply(len)

print(df["evidence_count"].describe())
print("\n具体分布：")
print(df["evidence_count"].value_counts().sort_index())


print("\n" + "=" * 60)
print("六、样本数量最多的公司")
print("=" * 60)

print(df["company"].value_counts().head(10))