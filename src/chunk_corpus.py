import json
from pathlib import Path
from statistics import mean, median

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evidence_page_corpus.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evidence_chunk_corpus.jsonl"
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# SentenceTransformer使用的最大序列长度
MODEL_MAX_TOKENS = 256

# 每个文本块最多包含的正文Token数
MAX_CONTENT_TOKENS = 240

# 相邻文本块重叠的Token数
TOKEN_OVERLAP = 40

# 在目标切分位置之前寻找自然边界的范围
BOUNDARY_SEARCH_TOKENS = 30


def load_jsonl(file_path):
    """读取JSONL文件。"""

    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON解析失败："
                    f"file={file_path}, "
                    f"line={line_number}"
                ) from error

    return records


def is_natural_boundary(text, offsets, token_index):
    """
    判断某个Token位置是否为较自然的文本边界。

    如果前一个Token和当前Token之间存在空格或换行，
    就认为可以在这里切分。
    """

    if token_index <= 0:
        return True

    if token_index >= len(offsets):
        return True

    previous_token_end = offsets[token_index - 1][1]
    current_token_start = offsets[token_index][0]

    text_between_tokens = text[
        previous_token_end:current_token_start
    ]

    return any(
        character.isspace()
        for character in text_between_tokens
    )


def choose_end_token(
    text,
    offsets,
    start_token,
):
    """
    选择当前文本块的结束Token。

    首先确定最多240个Token的位置，
    再尝试向前寻找空格或换行边界。
    """

    total_tokens = len(offsets)

    target_end_token = min(
        start_token + MAX_CONTENT_TOKENS,
        total_tokens,
    )

    if target_end_token >= total_tokens:
        return total_tokens

    minimum_search_token = max(
        start_token + MAX_CONTENT_TOKENS // 2,
        target_end_token - BOUNDARY_SEARCH_TOKENS,
    )

    for candidate_token in range(
        target_end_token,
        minimum_search_token,
        -1,
    ):
        if is_natural_boundary(
            text=text,
            offsets=offsets,
            token_index=candidate_token,
        ):
            return candidate_token

    # 如果没有找到合适边界，
    # 就使用原定的Token结束位置
    return target_end_token


def move_start_to_boundary(
    text,
    offsets,
    token_index,
    end_token,
):
    """
    如果下一个文本块起点位于单词中间，
    就向后移动到下一个自然边界。
    """

    while (
        token_index < end_token
        and not is_natural_boundary(
            text=text,
            offsets=offsets,
            token_index=token_index,
        )
    ):
        token_index += 1

    return token_index


def split_text_by_tokens(text, tokenizer):
    """
    使用向量模型对应的Tokenizer切分完整页面。

    返回每个文本块的：
    - 原始文本
    - 字符范围
    - Token范围
    - Token数量
    """

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"]

    if not offsets:
        return []

    chunks = []
    total_tokens = len(offsets)
    start_token = 0

    while start_token < total_tokens:
        end_token = choose_end_token(
            text=text,
            offsets=offsets,
            start_token=start_token,
        )

        start_char = offsets[start_token][0]
        end_char = offsets[end_token - 1][1]

        raw_chunk_text = text[start_char:end_char]

        # 记录strip前后删除的字符数，
        # 使字符位置仍然准确
        left_trim_count = (
            len(raw_chunk_text)
            - len(raw_chunk_text.lstrip())
        )

        right_trim_count = (
            len(raw_chunk_text)
            - len(raw_chunk_text.rstrip())
        )

        actual_start_char = (
            start_char + left_trim_count
        )

        actual_end_char = (
            end_char - right_trim_count
        )

        chunk_text = text[
            actual_start_char:actual_end_char
        ]

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "start_char": actual_start_char,
                    "end_char": actual_end_char,
                    "start_token": start_token,
                    "end_token": end_token,
                    "content_token_count": (
                        end_token - start_token
                    ),
                }
            )

        if end_token >= total_tokens:
            break

        next_start_token = (
            end_token - TOKEN_OVERLAP
        )

        next_start_token = move_start_to_boundary(
            text=text,
            offsets=offsets,
            token_index=next_start_token,
            end_token=end_token,
        )

        # 防止循环无法向前移动
        if next_start_token <= start_token:
            next_start_token = end_token

        start_token = next_start_token

    return chunks


def main():
    if TOKEN_OVERLAP >= MAX_CONTENT_TOKENS:
        raise ValueError(
            "TOKEN_OVERLAP必须小于"
            "MAX_CONTENT_TOKENS。"
        )

    print("=" * 70)
    print("加载Tokenizer")
    print("=" * 70)
    print(f"Tokenizer：{MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise ValueError(
            "当前Tokenizer不支持offset_mapping，"
            "必须使用Fast Tokenizer。"
        )

    special_token_count = (
        tokenizer.num_special_tokens_to_add(
            pair=False
        )
    )

    estimated_max_input_tokens = (
        MAX_CONTENT_TOKENS
        + special_token_count
    )

    if estimated_max_input_tokens > MODEL_MAX_TOKENS:
        raise ValueError(
            f"正文Token与特殊Token之和超过模型限制："
            f"{estimated_max_input_tokens}"
            f" > {MODEL_MAX_TOKENS}"
        )

    print(f"模型最大Token数：{MODEL_MAX_TOKENS}")
    print(f"正文最大Token数：{MAX_CONTENT_TOKENS}")
    print(f"重叠Token数：{TOKEN_OVERLAP}")
    print(f"特殊Token数：{special_token_count}")
    print(
        f"预计最大输入Token数："
        f"{estimated_max_input_tokens}"
    )

    # 完整页面通常很长，但这里仅进行Token切分，
    # 不会直接输入模型，所以关闭Tokenizer的长度警告
    tokenizer.model_max_length = 1_000_000_000

    pages = load_jsonl(INPUT_PATH)

    all_chunks = []

    for page in pages:
        page_chunks = split_text_by_tokens(
            text=page["text"],
            tokenizer=tokenizer,
        )

        if not page_chunks:
            raise ValueError(
                f"页面未生成文本块："
                f"{page['chunk_id']}"
            )

        for chunk_index, chunk in enumerate(
            page_chunks
        ):
            chunk_record = {
                "chunk_id": (
                    f"{page['chunk_id']}"
                    f"__chunk_{chunk_index:03d}"
                ),
                "parent_page_id": page["chunk_id"],
                "company": page["company"],
                "doc_name": page["doc_name"],
                "page_num": page["page_num"],
                "chunk_index": chunk_index,
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "start_token": chunk["start_token"],
                "end_token": chunk["end_token"],
                "content_token_count": chunk[
                    "content_token_count"
                ],
                "text": chunk["text"],

                # 仅供后续页面级检索评测使用，
                # 不能加入模型提示词或向量文本
                "source_question_ids": page[
                    "source_question_ids"
                ],
            }

            all_chunks.append(chunk_record)

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in all_chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "发现重复的chunk_id。"
        )

    token_counts = [
        chunk["content_token_count"]
        for chunk in all_chunks
    ]

    if max(token_counts) > MAX_CONTENT_TOKENS:
        raise ValueError(
            "生成的文本块超过设定的Token限制。"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for chunk in all_chunks:
            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )

    character_lengths = [
        len(chunk["text"])
        for chunk in all_chunks
    ]

    chunks_per_page = [
        sum(
            chunk["parent_page_id"]
            == page["chunk_id"]
            for chunk in all_chunks
        )
        for page in pages
    ]

    print("\n" + "=" * 70)
    print("FinanceBench Token级页面切块完成")
    print("=" * 70)
    print(f"输入页面数量：{len(pages)}")
    print(f"生成文本块数量：{len(all_chunks)}")
    print(
        f"每页平均文本块数："
        f"{mean(chunks_per_page):.2f}"
    )
    print(
        f"平均正文Token数："
        f"{mean(token_counts):.2f}"
    )
    print(
        f"正文Token数中位数："
        f"{median(token_counts):.2f}"
    )
    print(
        f"最少正文Token数："
        f"{min(token_counts)}"
    )
    print(
        f"最多正文Token数："
        f"{max(token_counts)}"
    )
    print(
        f"平均字符数："
        f"{mean(character_lengths):.2f}"
    )
    print(
        f"最长字符数："
        f"{max(character_lengths)}"
    )
    print(f"保存位置：{OUTPUT_PATH}")

    print("\n前两个文本块：")

    for chunk in all_chunks[:2]:
        print("-" * 70)
        print(f"chunk_id：{chunk['chunk_id']}")
        print(
            f"Token范围："
            f"{chunk['start_token']}"
            f"～{chunk['end_token']}"
        )
        print(
            f"正文Token数："
            f"{chunk['content_token_count']}"
        )
        print(
            f"字符范围："
            f"{chunk['start_char']}"
            f"～{chunk['end_char']}"
        )
        print(
            f"文本开头："
            f"{chunk['text'][:300]}"
        )


if __name__ == "__main__":
    main()