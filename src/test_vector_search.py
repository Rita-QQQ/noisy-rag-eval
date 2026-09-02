import hashlib
import json
from pathlib import Path

import numpy as np
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

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evidence_chunk_embeddings_meta.json"
)


# 先测试开发集中的第一道题
QUESTION_INDEX = 0

# 返回相似度最高的5个文本块
TOP_K = 5


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
    """计算语料中所有chunk_id的哈希值。"""

    chunk_id_text = "\n".join(
        record["chunk_id"]
        for record in records
    )

    return hashlib.sha256(
        chunk_id_text.encode("utf-8")
    ).hexdigest()


def get_gold_pages(sample):
    """
    获取当前问题对应的标准证据页面。

    一个问题可能有1～3个标准证据页。
    """

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
    print("加载向量检索数据")
    print("=" * 70)

    dev_samples = load_jsonl(DEV_PATH)
    corpus = load_jsonl(CORPUS_PATH)

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    # 检查问题索引是否有效
    if not 0 <= QUESTION_INDEX < len(dev_samples):
        raise IndexError(
            f"QUESTION_INDEX超出范围："
            f"{QUESTION_INDEX}"
        )

    # 检查语料数量与向量数量是否一致
    if embeddings.shape[0] != len(corpus):
        raise ValueError(
            "语料数量与向量数量不一致："
            f"corpus={len(corpus)}, "
            f"embeddings={embeddings.shape[0]}"
        )

    # 检查向量维度
    if embeddings.shape[1] != metadata[
        "embedding_dimension"
    ]:
        raise ValueError(
            "向量维度与元数据不一致。"
        )

    # 检查语料顺序是否发生变化
    current_chunk_id_hash = (
        calculate_chunk_id_hash(corpus)
    )

    if current_chunk_id_hash != metadata[
        "chunk_ids_sha256"
    ]:
        raise ValueError(
            "语料顺序与生成向量时不一致。"
            "请重新运行build_embeddings.py。"
        )

    sample = dev_samples[QUESTION_INDEX]

   # 兼容原始FinanceBench和处理后的开发集字段名
    question_id = (
        sample.get("financebench_id")
        or sample.get("sample_id")
        or sample.get("id")
    )

    if question_id is None:
        raise KeyError(
            "没有找到题目编号字段。"
            f"当前样本字段：{list(sample.keys())}"
        )

    question = sample["question"]
    gold_pages = get_gold_pages(sample)
    
    print(f"开发集样本数量：{len(dev_samples)}")
    print(f"语料块数量：{len(corpus)}")
    print(f"向量形状：{embeddings.shape}")
    print(f"使用模型：{metadata['model_name']}")

    print("\n" + "=" * 70)
    print("当前问题")
    print("=" * 70)
    print(f"样本编号：{question_id}")
    print(f"公司：{sample['company']}")
    print(f"问题类型：{sample['question_type']}")
    print(f"问题：{question}")

    print("\n标准证据页面：")

    for doc_name, page_num in sorted(gold_pages):
        print(
            f"- 文档：{doc_name}，"
            f"页码：{page_num}"
        )

    print("\n正在生成问题向量……")

    model = SentenceTransformer(
        metadata["model_name"]
    )

    query_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if query_embedding.shape[0] != embeddings.shape[1]:
        raise ValueError(
            "问题向量与语料向量的维度不一致。"
        )

    # 因为问题向量和语料向量都已经归一化，
    # 点积就等于余弦相似度
    similarity_scores = (
        embeddings @ query_embedding
    )

    # 按照相似度从高到低排序
    top_indices = np.argsort(
        similarity_scores
    )[::-1][:TOP_K]

    hit_ranks = []

    print("\n" + "=" * 70)
    print(f"向量检索Top-{TOP_K}结果")
    print("=" * 70)

    for rank, corpus_index in enumerate(
        top_indices,
        start=1,
    ):
        result = corpus[corpus_index]
        score = float(
            similarity_scores[corpus_index]
        )

        result_page = (
            result["doc_name"],
            int(result["page_num"]),
        )

        is_gold_page = (
            result_page in gold_pages
        )

        if is_gold_page:
            hit_ranks.append(rank)

        print("\n" + "-" * 70)
        print(f"排名：{rank}")
        print(f"相似度：{score:.4f}")
        print(f"命中标准证据页：{is_gold_page}")
        print(f"chunk_id：{result['chunk_id']}")
        print(f"公司：{result['company']}")
        print(f"文档：{result['doc_name']}")
        print(f"页码：{result['page_num']}")
        print(
            f"正文Token数："
            f"{result['content_token_count']}"
        )
        print("文本：")
        print(result["text"][:800])

    print("\n" + "=" * 70)
    print("单题检索结论")
    print("=" * 70)

    if hit_ranks:
        first_hit_rank = min(hit_ranks)

        print("Top-5是否命中标准证据页：是")
        print(
            f"首次命中排名：{first_hit_rank}"
        )

        print(
            f"Hit@1："
            f"{int(first_hit_rank <= 1)}"
        )
        print(
            f"Hit@3："
            f"{int(first_hit_rank <= 3)}"
        )
        print(
            f"Hit@5："
            f"{int(first_hit_rank <= 5)}"
        )

    else:
        print("Top-5是否命中标准证据页：否")
        print("首次命中排名：无")
        print("Hit@1：0")
        print("Hit@3：0")
        print("Hit@5：0")


if __name__ == "__main__":
    main()