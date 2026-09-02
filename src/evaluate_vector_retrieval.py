import hashlib
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEV_PATH = (
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

RAW_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw_outputs"
    / "vector_retrieval_dev.jsonl"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "vector_retrieval_dev_summary.json"
)

BY_TYPE_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "vector_retrieval_dev_by_type.csv"
)


HIT_K_VALUES = [1, 3, 5, 10]

# 每道题保存前10个检索结果，方便错误分析
SAVE_TOP_K = 10


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


def calculate_chunk_id_hash(records):
    """计算语料顺序的哈希值。"""

    chunk_id_text = "\n".join(
        record["chunk_id"]
        for record in records
    )

    return hashlib.sha256(
        chunk_id_text.encode("utf-8")
    ).hexdigest()


def get_question_id(sample):
    """兼容不同的题目编号字段。"""

    question_id = (
        sample.get("financebench_id")
        or sample.get("sample_id")
        or sample.get("id")
    )

    if question_id is None:
        raise KeyError(
            "没有找到题目编号字段。"
            f"实际字段：{list(sample.keys())}"
        )

    return question_id


def get_gold_pages(sample):
    """获取一道题的全部标准证据页。"""

    if "evidence" not in sample:
        raise KeyError(
            "样本中没有evidence字段。"
            f"实际字段：{list(sample.keys())}"
        )

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

    if not gold_pages:
        raise ValueError(
            f"样本没有标准证据页："
            f"{get_question_id(sample)}"
        )

    return gold_pages


def page_to_dict(page):
    """把页面元组转换成可保存的字典。"""

    doc_name, page_num = page

    return {
        "doc_name": doc_name,
        "page_num": int(page_num),
    }


def main():
    print("=" * 70)
    print("加载向量检索评测数据")
    print("=" * 70)

    dev_samples = load_jsonl(DEV_PATH)
    corpus = load_jsonl(CORPUS_PATH)
    corpus_embeddings = np.load(EMBEDDINGS_PATH)

    with EMBEDDINGS_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        embeddings_metadata = json.load(file)

    if corpus_embeddings.shape[0] != len(corpus):
        raise ValueError(
            "语料数量与向量数量不一致："
            f"corpus={len(corpus)}, "
            f"embeddings={corpus_embeddings.shape[0]}"
        )

    if corpus_embeddings.shape[1] != embeddings_metadata[
        "embedding_dimension"
    ]:
        raise ValueError(
            "向量维度与元数据不一致。"
        )

    current_hash = calculate_chunk_id_hash(
        corpus
    )

    expected_hash = embeddings_metadata[
        "chunk_ids_sha256"
    ]

    if current_hash != expected_hash:
        raise ValueError(
            "当前语料与向量文件不对应。"
            "请重新运行build_embeddings.py。"
        )

    print(f"开发集问题数量：{len(dev_samples)}")
    print(f"文本块数量：{len(corpus)}")
    print(f"语料向量形状：{corpus_embeddings.shape}")
    print(
        f"向量模型："
        f"{embeddings_metadata['model_name']}"
    )

    questions = [
        sample["question"]
        for sample in dev_samples
    ]

    print("\n正在加载向量模型……")

    model = SentenceTransformer(
        embeddings_metadata["model_name"]
    )

    print("正在批量生成30个问题向量……")

    query_embeddings = model.encode(
        questions,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 形状：
    # (30, 384) @ (384, 615)
    # 最终得到 (30, 615) 的相似度矩阵
    similarity_matrix = (
        query_embeddings
        @ corpus_embeddings.T
    )

    all_results = []

    print("\n正在评测每道问题的检索结果……")

    for sample_index, sample in enumerate(
        dev_samples
    ):
        question_id = get_question_id(sample)
        gold_pages = get_gold_pages(sample)

        scores = similarity_matrix[sample_index]

        ranked_indices = np.argsort(
            scores
        )[::-1]

        first_gold_rank = None

        for rank, corpus_index in enumerate(
            ranked_indices,
            start=1,
        ):
            candidate = corpus[corpus_index]

            candidate_page = (
                candidate["doc_name"],
                int(candidate["page_num"]),
            )

            if candidate_page in gold_pages:
                first_gold_rank = rank
                break

        if first_gold_rank is None:
            reciprocal_rank = 0.0
        else:
            reciprocal_rank = (
                1.0 / first_gold_rank
            )

        result_record = {
            "sample_id": question_id,
            "company": sample["company"],
            "question_type": sample[
                "question_type"
            ],
            "question": sample["question"],
            "gold_pages": [
                page_to_dict(page)
                for page in sorted(gold_pages)
            ],
            "gold_page_count": len(gold_pages),
            "first_gold_rank": first_gold_rank,
            "reciprocal_rank": reciprocal_rank,
        }

        for k in HIT_K_VALUES:
            result_record[f"hit_at_{k}"] = int(
                first_gold_rank is not None
                and first_gold_rank <= k
            )

            top_k_pages = {
                (
                    corpus[index]["doc_name"],
                    int(corpus[index]["page_num"]),
                )
                for index in ranked_indices[:k]
            }

            retrieved_gold_pages = (
                top_k_pages & gold_pages
            )

            result_record[
                f"gold_page_recall_at_{k}"
            ] = (
                len(retrieved_gold_pages)
                / len(gold_pages)
            )

        top_results = []

        for rank, corpus_index in enumerate(
            ranked_indices[:SAVE_TOP_K],
            start=1,
        ):
            candidate = corpus[corpus_index]

            candidate_page = (
                candidate["doc_name"],
                int(candidate["page_num"]),
            )

            top_results.append(
                {
                    "rank": rank,
                    "score": float(
                        scores[corpus_index]
                    ),
                    "chunk_id": candidate[
                        "chunk_id"
                    ],
                    "parent_page_id": candidate[
                        "parent_page_id"
                    ],
                    "company": candidate[
                        "company"
                    ],
                    "doc_name": candidate[
                        "doc_name"
                    ],
                    "page_num": int(
                        candidate["page_num"]
                    ),
                    "is_gold_page": (
                        candidate_page
                        in gold_pages
                    ),
                    "text_preview": candidate[
                        "text"
                    ][:500],
                }
            )

        result_record["top_results"] = (
            top_results
        )

        all_results.append(result_record)

    RAW_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RAW_RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for result in all_results:
            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    first_gold_ranks = [
        result["first_gold_rank"]
        for result in all_results
        if result["first_gold_rank"] is not None
    ]

    summary = {
        "system": "dense_vector_retrieval",
        "embedding_model": embeddings_metadata[
            "model_name"
        ],
        "total_samples": len(all_results),
        "corpus_chunk_count": len(corpus),
        "mrr": mean(
            result["reciprocal_rank"]
            for result in all_results
        ),
        "mean_first_gold_rank": mean(
            first_gold_ranks
        ),
        "median_first_gold_rank": median(
            first_gold_ranks
        ),
    }

    for k in HIT_K_VALUES:
        summary[f"hit_at_{k}"] = mean(
            result[f"hit_at_{k}"]
            for result in all_results
        )

        summary[
            f"mean_gold_page_recall_at_{k}"
        ] = mean(
            result[f"gold_page_recall_at_{k}"]
            for result in all_results
        )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    result_dataframe = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key not in {
                    "gold_pages",
                    "top_results",
                }
            }
            for result in all_results
        ]
    )

    aggregation = {
        "sample_id": "count",
        "reciprocal_rank": "mean",
        "first_gold_rank": "mean",
    }

    for k in HIT_K_VALUES:
        aggregation[f"hit_at_{k}"] = "mean"
        aggregation[
            f"gold_page_recall_at_{k}"
        ] = "mean"

    by_type = (
        result_dataframe
        .groupby(
            "question_type",
            as_index=False,
        )
        .agg(aggregation)
        .rename(
            columns={
                "sample_id": "sample_count",
                "reciprocal_rank": "mrr",
                "first_gold_rank": (
                    "mean_first_gold_rank"
                ),
            }
        )
    )

    by_type.to_csv(
        BY_TYPE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("开发集向量检索评测结果")
    print("=" * 70)
    print(
        f"样本数量："
        f"{summary['total_samples']}"
    )
    print(
        f"Hit@1："
        f"{summary['hit_at_1']:.2%}"
    )
    print(
        f"Hit@3："
        f"{summary['hit_at_3']:.2%}"
    )
    print(
        f"Hit@5："
        f"{summary['hit_at_5']:.2%}"
    )
    print(
        f"Hit@10："
        f"{summary['hit_at_10']:.2%}"
    )
    print(f"MRR：{summary['mrr']:.4f}")
    print(
        f"首次命中平均排名："
        f"{summary['mean_first_gold_rank']:.2f}"
    )
    print(
        f"首次命中排名中位数："
        f"{summary['median_first_gold_rank']:.2f}"
    )

    print("\n按问题类型统计：")
    print(by_type.to_string(index=False))

    print("\n结果已保存：")
    print(RAW_RESULTS_PATH)
    print(SUMMARY_PATH)
    print(BY_TYPE_PATH)


if __name__ == "__main__":
    main()