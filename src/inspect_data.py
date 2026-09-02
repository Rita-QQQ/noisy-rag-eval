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
print(f"数据文件：{DATA_PATH}")
print(f"样本数量：{len(df)}")
print(f"字段数量：{len(df.columns)}")
print(f"全部字段：{df.columns.tolist()}")
print("=" * 60)

sample = df.iloc[0]

print(f"样本编号：{sample['financebench_id']}")
print(f"公司：{sample['company']}")
print(f"文档：{sample['doc_name']}")
print(f"问题：{sample['question']}")
print(f"标准答案：{sample['answer']}")
print(f"问题类型：{sample['question_type']}")
print(f"推理类型：{sample['question_reasoning']}")

# 一道题可能对应一条或多条标准证据
evidence_list = sample["evidence"]

print(f"证据数量：{len(evidence_list)}")

if evidence_list:
    first_evidence = evidence_list[0]

    print(f"证据包含的字段：{list(first_evidence.keys())}")
    print(f"证据文本：{first_evidence.get('evidence_text')}")
    print(
        "证据文档：",
        first_evidence.get(
            "evidence_doc_name",
            first_evidence.get("doc_name")
        ),
    )
    print(f"证据页码：{first_evidence.get('evidence_page_num')}")