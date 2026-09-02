import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


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

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evidence_chunk_embeddings.npy"
)

EMBEDDINGS_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evidence_chunk_embeddings_meta.json"
)


# 当前仍然检查开发集第一道题
QUESTION_INDEX = 0


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


def calculate_chunk_id_hash(records):
    """计算语料顺序哈希值。"""

    chunk_id_text = "\n".join(
        record["chunk_id"]
        for record in records
    )

    return hashlib.sha256(
        chunk_id_text.encode("utf-8")
    ).hexdigest()


def normalize_text(text):
    """
    统一大小写和空白符。

    例如：
    'Future   payment\\namount'
    会变成：
    'future payment amount'
    """

    return " ".join(
        str(text).lower().split()
    )


def tokenize_for_overlap(text):
    """
    用于计算证据词语覆盖率。

    保留英文字母、数字、小数和百分数，
    忽略换行及大部分标点差异。
    """

    normalized = normalize_text(text)

    return re.findall(
        r"[a-z]+|\d+(?:\.\d+)?%?",
        normalized,
    )


def calculate_evidence_coverage(
    evidence_text,
    chunk_text,
):
    """
    计算一个chunk覆盖了多少标准证据Token。

    coverage = 标准证据中被chunk覆盖的Token数
               / 标准证据Token总数
    """

    evidence_tokens = tokenize_for_overlap(
        evidence_text
    )

    chunk_tokens = tokenize_for_overlap(
        chunk_text
    )

    if not evidence_tokens:
        return 0.0

    evidence_counter = Counter(
        evidence_tokens
    )

    chunk_counter = Counter(
        chunk_tokens
    )

    overlap_counter = (
        evidence_counter
        & chunk_counter
    )

    overlap_count = sum(
        overlap_counter.values()
    )

    return (
        overlap_count
        / len(evidence_tokens)
    )


def get_gold_pages(sample):
    """获取样本的标准证据页面。"""

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
    print("标准证据Chunk排名诊断")
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

    corpus_embeddings = np.load(
        EMBEDDINGS_PATH
    )

    with EMBEDDINGS_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if corpus_embeddings.shape[0] != len(corpus):
        raise ValueError(
            "语料数量与向量数量不一致。"
        )

    if corpus_embeddings.ndim != 2:
        raise ValueError(
            f"语料向量形状异常："
            f"{corpus_embeddings.shape}"
        )

    if corpus_embeddings.shape[1] != (
        metadata["embedding_dimension"]
    ):
        raise ValueError(
            "语料向量维度与元数据不一致。"
        )

    current_hash = calculate_chunk_id_hash(
        corpus
    )

    if current_hash != metadata[
        "chunk_ids_sha256"
    ]:
        raise ValueError(
            "语料与向量文件不对应，"
            "请重新运行build_embeddings.py。"
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
    print(f"问题类型：{sample['question_type']}")
    print(f"问题：{question}")
    print(f"标准答案：{sample['gold_answer']}")

    print("\n标准证据页面：")

    for doc_name, page_num in sorted(
        gold_pages
    ):
        print(
            f"- 文档：{doc_name}，"
            f"页码：{page_num}"
        )

    print("\n" + "=" * 70)
    print("FinanceBench标准局部证据")
    print("=" * 70)

    for evidence_index, evidence_text in enumerate(
        gold_evidence_texts,
        start=1,
    ):
        print("\n" + "-" * 70)
        print(f"标准证据编号：G{evidence_index}")
        print(evidence_text)

    print("\n正在计算全部Chunk的检索排名……")

    model = SentenceTransformer(
        metadata["model_name"]
    )

    query_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    similarity_scores = (
        corpus_embeddings
        @ query_embedding
    )

    ranked_indices = np.argsort(
        similarity_scores
    )[::-1]

    # 建立：
    # corpus索引 -> 全局检索排名
    global_rank_by_index = {
        int(corpus_index): rank
        for rank, corpus_index in enumerate(
            ranked_indices,
            start=1,
        )
    }

    gold_page_chunks = []

    for corpus_index, chunk in enumerate(
        corpus
    ):
        chunk_page = (
            chunk["doc_name"],
            int(chunk["page_num"]),
        )

        if chunk_page not in gold_pages:
            continue

        normalized_chunk = normalize_text(
            chunk["text"]
        )

        exact_contains_list = [
            normalize_text(evidence_text)
            in normalized_chunk
            for evidence_text
            in gold_evidence_texts
        ]

        coverage_list = [
            calculate_evidence_coverage(
                evidence_text=evidence_text,
                chunk_text=chunk["text"],
            )
            for evidence_text
            in gold_evidence_texts
        ]

        gold_page_chunks.append(
            {
                "global_rank": (
                    global_rank_by_index[
                        corpus_index
                    ]
                ),
                "score": float(
                    similarity_scores[
                        corpus_index
                    ]
                ),
                "chunk_id": chunk["chunk_id"],
                "doc_name": chunk["doc_name"],
                "page_num": int(
                    chunk["page_num"]
                ),
                "content_token_count": chunk[
                    "content_token_count"
                ],
                "exact_contains": any(
                    exact_contains_list
                ),
                "maximum_coverage": max(
                    coverage_list,
                    default=0.0,
                ),
                "text": chunk["text"],
            }
        )

    gold_page_chunks.sort(
        key=lambda result: result[
            "global_rank"
        ]
    )

    print("\n" + "=" * 70)
    print("标准证据页中所有Chunk的全局排名")
    print("=" * 70)

    for result in gold_page_chunks:
        print("\n" + "-" * 70)
        print(
            f"全局检索排名："
            f"{result['global_rank']}"
        )
        print(
            f"相似度："
            f"{result['score']:.4f}"
        )
        print(
            f"chunk_id："
            f"{result['chunk_id']}"
        )
        print(
            f"文档：{result['doc_name']}"
        )
        print(
            f"页码：{result['page_num']}"
        )
        print(
            f"正文Token数："
            f"{result['content_token_count']}"
        )
        print(
            f"是否完整包含标准证据："
            f"{result['exact_contains']}"
        )
        print(
            f"最高证据Token覆盖率："
            f"{result['maximum_coverage']:.2%}"
        )
        print("文本：")
        print(result["text"])

    if not gold_page_chunks:
        raise ValueError(
            "语料库中没有找到标准证据页。"
        )

    best_overlap_chunk = max(
        gold_page_chunks,
        key=lambda result: result[
            "maximum_coverage"
        ],
    )

    exact_chunks = [
        result
        for result in gold_page_chunks
        if result["exact_contains"]
    ]

    print("\n" + "=" * 70)
    print("诊断结论")
    print("=" * 70)

    if exact_chunks:
        best_exact_rank = min(
            result["global_rank"]
            for result in exact_chunks
        )

        print(
            "存在完整包含标准局部证据的Chunk。"
        )
        print(
            f"最佳精确证据Chunk排名："
            f"{best_exact_rank}"
        )
        print(
            f"精确Evidence Hit@5："
            f"{int(best_exact_rank <= 5)}"
        )

    else:
        print(
            "没有单个Chunk完整包含标准局部证据。"
        )
        print(
            "可能原因：标准证据跨越多个Chunk，"
            "或者文本格式存在差异。"
        )

    print(
        f"最高覆盖率Chunk："
        f"{best_overlap_chunk['chunk_id']}"
    )
    print(
        f"该Chunk全局排名："
        f"{best_overlap_chunk['global_rank']}"
    )
    print(
        f"最高覆盖率："
        f"{best_overlap_chunk['maximum_coverage']:.2%}"
    )


if __name__ == "__main__":
    main()