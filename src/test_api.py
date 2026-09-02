import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# 找到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 从项目根目录的.env文件中加载配置
load_dotenv(PROJECT_ROOT / ".env")


# 读取环境变量
api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
model = os.getenv("LLM_MODEL")


# 检查配置是否存在，但绝不打印API Key
if not api_key:
    raise ValueError("没有读取到LLM_API_KEY，请检查.env文件")

if not base_url:
    raise ValueError("没有读取到LLM_BASE_URL，请检查.env文件")

if not model:
    raise ValueError("没有读取到LLM_MODEL，请检查.env文件")


# 创建DeepSeek客户端
# DeepSeek兼容OpenAI SDK，因此可以使用OpenAI这个Python类
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)


print(f"正在测试模型：{model}")
print(f"API地址：{base_url}")


# 发送一个极短的测试请求
response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "Follow the user's instruction exactly.",
        },
        {
            "role": "user",
            "content": "Reply with exactly: API connection successful",
        },
    ],
    temperature=0,
    max_tokens=20,

    # DeepSeek V4默认开启思考模式；
    # 基线实验暂时关闭思考模式，减少费用并保持条件稳定
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    },
)


answer = response.choices[0].message.content

print("\n模型回答：")
print(answer)


# usage记录本次请求使用了多少token
if response.usage:
    print("\nToken使用情况：")
    print(f"输入Token：{response.usage.prompt_tokens}")
    print(f"输出Token：{response.usage.completion_tokens}")
    print(f"总Token：{response.usage.total_tokens}")