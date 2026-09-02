import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 加载.env中的配置
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL_NAME = os.getenv("LLM_MODEL")


if not API_KEY:
    raise ValueError("没有读取到LLM_API_KEY")

if not BASE_URL:
    raise ValueError("没有读取到LLM_BASE_URL")

if not MODEL_NAME:
    raise ValueError("没有读取到LLM_MODEL")


# 程序启动时创建一次客户端，之后重复使用
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


def call_llm_json(
    messages: list[dict[str, str]],
    max_tokens: int = 400,
) -> tuple[dict[str, Any], dict[str, int | None]]:
    """
    调用大模型，并要求模型返回JSON。

    参数：
        messages：发送给模型的对话消息
        max_tokens：允许模型输出的最大Token数

    返回：
        result：解析后的JSON字典
        usage：本次请求的Token使用情况
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,

        # 要求API返回合法JSON
        response_format={
            "type": "json_object"
        },

        # 普通基线关闭思考模式
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("模型返回了空内容")

    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"模型返回的内容不是合法JSON：\n{content}"
        ) from error

    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    else:
        usage = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

    return result, usage