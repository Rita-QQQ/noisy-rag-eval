from pathlib import Path

import pandas as pd

from llm_client import MODEL_NAME, call_llm_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEV_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dev_30.jsonl"
)


from experiment_protocol import (
    PROMPT_VERSION,
    QAAnswer,
    build_system_prompt,
)


# 读取30条开发集
dev_df = pd.read_json(DEV_DATA_PATH, lines=True)

# 暂时只取第一道题
sample = dev_df.iloc[0]


system_prompt = build_system_prompt(
    "llm_only"
)


user_prompt = f"""
Company:
{sample["company"]}

Question:
{sample["question"]}
""".strip()


messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": user_prompt,
    },
]


# 调用DeepSeek API
raw_result, usage = call_llm_json(
    messages,
    max_tokens=600,
)


# 用Pydantic检查回答字段是否正确
validated_result = QAAnswer.model_validate(
    raw_result
)


print("=" * 70)
print("LLM Only：第一道开发集问题")
print("=" * 70)

print(f"样本编号：{sample['sample_id']}")
print(f"问题类型：{sample['question_type']}")
print(f"公司：{sample['company']}")
print(f"问题：{sample['question']}")

print("\n模型回答：")
print(validated_result.model_dump_json(indent=2))

print("\n标准答案：")
print(sample["gold_answer"])

print("\nToken使用情况：")
print(usage)

print("\n注意：标准答案只在模型回答之后显示，没有发送给模型。")
print(f"当前模型：{MODEL_NAME}")
print(f"Prompt版本：{PROMPT_VERSION}")