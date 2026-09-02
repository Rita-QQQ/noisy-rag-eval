import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEV_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dev_30.jsonl"
)

CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evidence_chunk_corpus.jsonl"
)


QUESTION_INDEX = 0
TOP_K = 10


# 去掉对金融检索帮助较小的常见词
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "much",
    "of",
    "on",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
}


def load_jsonl(file_path):
    """读取JSONL文件。"""

    records = []

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON解析失败："
                    f"file={file_path}, "
                    f"line={line_number}"
                ) from error

    return records


def normalize_text(text):
    """统一大小写和空白符。"""

    return " ".join(
        str(text).lower().split()
    )


def tokenize_for_bm25(text):
    """
    把文本转换成BM25使用的词语列表。

    处理规则：
    - 转为小写；
    - 将$转换为usd；
    - 将&转换为and；
    - 保留英文单词、数字、小数和百分数；
    - 去除常见停用词。
    """

    text = str(text).lower()

    text = text.replace(
        "$",
        " usd ",
    )

    text = text.replace(
        "&",
        " and ",
    )

    tokens = re.findall(
        r"[a-z]+|\d+(?:\.\d+)?%?",
        text,
    )

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


def get_gold_pages(sample):
    """获取标准证据页面。"""

    gold_pages = set()

    for evidence in sample["evidence"]:
        doc_name = (
            evidence.get("doc_name")
            or sample["doc_name"]
        )

        page_num = int(
            evidence["evidence_page_num"]
        )

        gold_pages.add(
            (doc_name, page_num)
        )

    return gold_pages


def main():
    print("=" * 70)
    print("BM25单题检索测试")
    print("=" * 70)

    dev_df = pd.read_json(
        DEV_DATA_PATH,
        lines=True,
    )

    sample = dev_df.iloc[
        QUESTION_INDEX
    ].to_dict()

    corpus = load_jsonl(
        CORPUS_PATH
    )

    question = sample["question"]
    gold_pages = get_gold_pages(sample)

    gold_evidence_texts = [
        evidence["evidence_text"]
        for evidence in sample["evidence"]
        if evidence.get("evidence_text")
    ]

    print(f"样本编号：{sample['sample_id']}")
    print(f"公司：{sample['company']}")
    print(f"问题：{question}")
    print(f"标准答案：{sample['gold_answer']}")

    print("\n正在对615个文本块进行分词……")

    tokenized_corpus = [
        tokenize_for_bm25(
            chunk["text"]
        )
        for chunk in corpus
    ]

    if any(
        len(tokens) == 0
        for tokens in tokenized_corpus
    ):
        empty_count = sum(
            len(tokens) == 0
            for tokens in tokenized_corpus
        )

        raise ValueError(
            f"发现空的BM25文本块：{empty_count}"
        )

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    query_tokens = tokenize_for_bm25(
        question
    )

    print(f"问题分词：{query_tokens}")

    bm25_scores = np.asarray(
        bm25.get_scores(query_tokens),
        dtype=np.float32,
    )

    ranked_indices = np.argsort(
        bm25_scores
    )[::-1]

    first_gold_page_rank = None
    first_exact_evidence_rank = None

    print("\n" + "=" * 70)
    print(f"BM25 Top-{TOP_K}结果")
    print("=" * 70)

    for rank, corpus_index in enumerate(
        ranked_indices,
        start=1,
    ):
        chunk = corpus[corpus_index]

        chunk_page = (
            chunk["doc_name"],
            int(chunk["page_num"]),
        )

        is_gold_page = (
            chunk_page in gold_pages
        )

        normalized_chunk = normalize_text(
            chunk["text"]
        )

        contains_exact_evidence = any(
            normalize_text(evidence_text)
            in normalized_chunk
            for evidence_text
            in gold_evidence_texts
        )

        if (
            is_gold_page
            and first_gold_page_rank is None
        ):
            first_gold_page_rank = rank

        if (
            contains_exact_evidence
            and first_exact_evidence_rank is None
        ):
            first_exact_evidence_rank = rank

        if rank <= TOP_K:
            print("\n" + "-" * 70)
            print(f"排名：{rank}")
            print(
                f"BM25分数："
                f"{bm25_scores[corpus_index]:.4f}"
            )
            print(
                f"是否为标准证据页："
                f"{is_gold_page}"
            )
            print(
                f"是否完整包含标准证据："
                f"{contains_exact_evidence}"
            )
            print(
                f"chunk_id："
                f"{chunk['chunk_id']}"
            )
            print(
                f"文档：{chunk['doc_name']}"
            )
            print(
                f"页码：{chunk['page_num']}"
            )
            print("文本开头：")
            print(chunk["text"][:600])

    print("\n" + "=" * 70)
    print("BM25单题结论")
    print("=" * 70)
    print(
        f"首次标准页面排名："
        f"{first_gold_page_rank}"
    )
    print(
        f"首次精确证据Chunk排名："
        f"{first_exact_evidence_rank}"
    )

    if first_exact_evidence_rank is None:
        print("Exact Evidence Hit@5：0")
        print(
            "没有任何单个Chunk完整包含标准证据。"
        )
    else:
        print(
            f"Exact Evidence Hit@5："
            f"{int(first_exact_evidence_rank <= 5)}"
        )
        print(
            f"Exact Evidence Hit@10："
            f"{int(first_exact_evidence_rank <= 10)}"
        )


if __name__ == "__main__":
    main()