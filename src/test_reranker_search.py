import json

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)

from test_hybrid_search import (
    CORPUS_PATH,
    DEV_DATA_PATH,
    EMBEDDINGS_PATH,
    METADATA_PATH,
    build_rank_array,
    calculate_chunk_id_hash,
    get_gold_pages,
    load_jsonl,
    normalize_text,
    tokenize_for_bm25,
)


QUESTION_INDEX = 0

# Dense和BM25各自召回前20个候选
CANDIDATE_K = 20

# 重排后展示前10个
OUTPUT_TOP_K = 10

RERANKER_MODEL_NAME = (
    "cross-encoder/"
    "ms-marco-MiniLM-L6-v2"
)


def main():
    print("=" * 70)
    print("CrossEncoder二阶段重排测试")
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
            f"语料向量形状异常："
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
            "当前语料和向量文件不对应。"
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

    # =========================================================
    # 第一阶段A：Dense召回
    # =========================================================

    print("\n正在计算Dense Top-20……")

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

    # =========================================================
    # 第一阶段B：BM25召回
    # =========================================================

    print("正在计算BM25 Top-20……")

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

    # =========================================================
    # 合并两种方法的Top-20候选
    # =========================================================

    candidate_indices = sorted(
        set(
            dense_order[:CANDIDATE_K].tolist()
        )
        | set(
            bm25_order[:CANDIDATE_K].tolist()
        )
    )

    print(
        f"合并去重后的候选数量："
        f"{len(candidate_indices)}"
    )

    # 检查精确证据是否进入了候选池
    exact_evidence_candidate_indices = []

    for corpus_index in candidate_indices:
        normalized_chunk = normalize_text(
            corpus[corpus_index]["text"]
        )

        contains_exact_evidence = any(
            normalize_text(evidence_text)
            in normalized_chunk
            for evidence_text
            in gold_evidence_texts
        )

        if contains_exact_evidence:
            exact_evidence_candidate_indices.append(
                corpus_index
            )

    candidate_recall = bool(
        exact_evidence_candidate_indices
    )

    print(
        f"候选池是否包含精确证据："
        f"{candidate_recall}"
    )

    if not candidate_recall:
        print(
            "警告：第一阶段没有召回精确证据，"
            "重排器无法恢复未进入候选池的文本。"
        )

    # =========================================================
    # 第二阶段：CrossEncoder重排
    # =========================================================

    print(
        f"\n正在加载重排模型："
        f"{RERANKER_MODEL_NAME}"
    )

    reranker = CrossEncoder(
        RERANKER_MODEL_NAME,
        max_length=512,
    )

    reranker_pairs = []

    for corpus_index in candidate_indices:
        chunk = corpus[corpus_index]

        # 将可用来源元数据一同交给重排模型。
        # 不包含gold_pages或source_question_ids。
        candidate_text = f"""
Company: {chunk["company"]}
Document: {chunk["doc_name"]}
Page: {chunk["page_num"]}
Content:
{chunk["text"]}
""".strip()

        reranker_pairs.append(
            (
                question,
                candidate_text,
            )
        )

    reranker_scores = reranker.predict(
        reranker_pairs,
        batch_size=16,
        show_progress_bar=True,
    )

    reranker_scores = np.asarray(
        reranker_scores,
        dtype=np.float32,
    ).reshape(-1)

    if len(reranker_scores) != len(
        candidate_indices
    ):
        raise ValueError(
            "重排分数数量与候选数量不一致。"
        )

    local_reranked_order = np.argsort(
        reranker_scores
    )[::-1]

    reranked_indices = [
        candidate_indices[local_index]
        for local_index in local_reranked_order
    ]

    first_gold_page_rank = None
    first_exact_evidence_rank = None

    print("\n" + "=" * 70)
    print(
        f"CrossEncoder重排Top-"
        f"{OUTPUT_TOP_K}"
    )
    print("=" * 70)

    for reranked_rank, local_index in enumerate(
        local_reranked_order,
        start=1,
    ):
        corpus_index = candidate_indices[
            local_index
        ]

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
                reranked_rank
            )

        if (
            contains_exact_evidence
            and first_exact_evidence_rank
            is None
        ):
            first_exact_evidence_rank = (
                reranked_rank
            )

        if reranked_rank <= OUTPUT_TOP_K:
            print("\n" + "-" * 70)
            print(
                f"重排后排名："
                f"{reranked_rank}"
            )
            print(
                f"CrossEncoder分数："
                f"{reranker_scores[local_index]:.4f}"
            )
            print(
                f"Dense原始排名："
                f"{dense_ranks[corpus_index]}"
            )
            print(
                f"BM25原始排名："
                f"{bm25_ranks[corpus_index]}"
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
            print(chunk["text"][:700])

    print("\n" + "=" * 70)
    print("CrossEncoder重排结论")
    print("=" * 70)
    print(
        f"第一阶段精确证据召回："
        f"{candidate_recall}"
    )
    print(
        f"重排后首次标准页面排名："
        f"{first_gold_page_rank}"
    )
    print(
        f"重排后首次精确证据排名："
        f"{first_exact_evidence_rank}"
    )

    if first_exact_evidence_rank is None:
        print("Reranked Exact Hit@5：0")
        print("Reranked Exact Hit@10：0")
    else:
        print(
            f"Reranked Exact Hit@5："
            f"{int(first_exact_evidence_rank <= 5)}"
        )
        print(
            f"Reranked Exact Hit@10："
            f"{int(first_exact_evidence_rank <= 10)}"
        )


if __name__ == "__main__":
    main()