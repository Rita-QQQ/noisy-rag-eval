import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
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

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evidence_chunk_embeddings_meta.json"
)


QUESTION_INDEX = 0
TOP_K = 10

# RRF中的排名平滑常数
RRF_K = 60


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
    """统一大小写和空白符。"""

    return " ".join(
        str(text).lower().split()
    )


def tokenize_for_bm25(text):
    """生成BM25使用的Token。"""

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


def build_rank_array(ranked_indices):
    """
    把排序结果转换成排名数组。

    例如：
    ranked_indices = [5, 2, 0, ...]

    表示：
    corpus[5]排名1
    corpus[2]排名2
    corpus[0]排名3
    """

    ranks = np.empty(
        len(ranked_indices),
        dtype=np.int32,
    )

    ranks[ranked_indices] = np.arange(
        1,
        len(ranked_indices) + 1,
    )

    return ranks


def main():
    print("=" * 70)
    print("Dense＋BM25混合检索测试")
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

    with METADATA_PATH.open(
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
            f"向量形状异常："
            f"{corpus_embeddings.shape}"
        )

    if corpus_embeddings.shape[1] != (
        metadata["embedding_dimension"]
    ):
        raise ValueError(
            "向量维度与元数据不一致。"
        )

    current_hash = calculate_chunk_id_hash(
        corpus
    )

    if current_hash != metadata[
        "chunk_ids_sha256"
    ]:
        raise ValueError(
            "当前语料与向量文件不对应。"
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

    # Dense向量检索
    print("\n正在计算Dense排名……")

    embedding_model = SentenceTransformer(
        metadata["model_name"]
    )

    query_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dense_scores = (
        corpus_embeddings @ query_embedding
    )

    dense_order = np.argsort(
        dense_scores
    )[::-1]

    dense_ranks = build_rank_array(
        dense_order
    )

    # BM25检索
    print("正在计算BM25排名……")

    tokenized_corpus = [
        tokenize_for_bm25(
            chunk["text"]
        )
        for chunk in corpus
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    query_tokens = tokenize_for_bm25(
        question
    )

    bm25_scores = np.asarray(
        bm25.get_scores(query_tokens),
        dtype=np.float32,
    )

    bm25_order = np.argsort(
        bm25_scores
    )[::-1]

    bm25_ranks = build_rank_array(
        bm25_order
    )

    # Reciprocal Rank Fusion
    rrf_scores = (
        1.0 / (RRF_K + dense_ranks)
        + 1.0 / (RRF_K + bm25_ranks)
    )

    hybrid_order = np.argsort(
        rrf_scores
    )[::-1]

    first_gold_page_rank = None
    first_exact_evidence_rank = None

    print("\n" + "=" * 70)
    print(f"混合检索Top-{TOP_K}结果")
    print("=" * 70)

    for hybrid_rank, corpus_index in enumerate(
        hybrid_order,
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
            first_gold_page_rank = (
                hybrid_rank
            )

        if (
            contains_exact_evidence
            and first_exact_evidence_rank
            is None
        ):
            first_exact_evidence_rank = (
                hybrid_rank
            )

        if hybrid_rank <= TOP_K:
            print("\n" + "-" * 70)
            print(
                f"混合排名：{hybrid_rank}"
            )
            print(
                f"Dense排名："
                f"{dense_ranks[corpus_index]}"
            )
            print(
                f"BM25排名："
                f"{bm25_ranks[corpus_index]}"
            )
            print(
                f"RRF分数："
                f"{rrf_scores[corpus_index]:.6f}"
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
    print("混合检索单题结论")
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
        print("Exact Evidence Hit@10：0")
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