import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from experiment_protocol import (
    PROMPT_VERSION,
    QAAnswer,
    build_system_prompt,
)
from llm_client import (
    MODEL_NAME as LLM_MODEL_NAME,
    call_llm_json,
)


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


QUESTION_INDEX = 0
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
    """计算语料顺序的哈希值。"""

    chunk_id_text = "\n".join(
        record["chunk_id"]
        for record in records
    )

    return hashlib.sha256(
        chunk_id_text.encode("utf-8")
    ).hexdigest()


def get_gold_pages(sample):
    """
    获取标准证据页面。

    仅用于模型回答之后的实验检查，
    不会放进模型提示词。
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
    print("基础RAG：第一道开发集问题")
    print("=" * 70)

    # 读取开发集
    dev_df = pd.read_json(
        DEV_DATA_PATH,
        lines=True,
    )

    sample = dev_df.iloc[
        QUESTION_INDEX
    ].to_dict()

    # 读取检索语料和已经生成的向量
    corpus = load_jsonl(CORPUS_PATH)

    corpus_embeddings = np.load(
        EMBEDDINGS_PATH
    )

    with EMBEDDINGS_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        embeddings_metadata = json.load(file)

    # 检查语料和向量是否仍然对应
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
        embeddings_metadata["embedding_dimension"]
    ):
        raise ValueError(
            "语料向量维度与元数据不一致："
            f"embeddings={corpus_embeddings.shape[1]}, "
            f"metadata="
            f"{embeddings_metadata['embedding_dimension']}"
        )

    current_hash = calculate_chunk_id_hash(
        corpus
    )

    if current_hash != embeddings_metadata[
        "chunk_ids_sha256"
    ]:
        raise ValueError(
            "语料顺序与向量文件不一致。"
            "请重新运行build_embeddings.py。"
        )

    question = sample["question"]

    print(f"样本编号：{sample['sample_id']}")
    print(f"问题类型：{sample['question_type']}")
    print(f"公司：{sample['company']}")
    print(f"问题：{question}")

    print(
        f"\n正在进行Top-{TOP_K}向量检索……"
    )

    embedding_model = SentenceTransformer(
        embeddings_metadata["model_name"]
    )

    query_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    similarity_scores = (
        corpus_embeddings @ query_embedding
    )

    top_indices = np.argsort(
        similarity_scores
    )[::-1][:TOP_K]

    retrieved_results = []
    evidence_sections = []

    for rank, corpus_index in enumerate(
        top_indices,
        start=1,
    ):
        chunk = corpus[corpus_index]
        citation_id = f"E{rank}"

        retrieved_result = {
            "citation_id": citation_id,
            "rank": rank,
            "score": float(
                similarity_scores[corpus_index]
            ),
            "chunk_id": chunk["chunk_id"],
            "company": chunk["company"],
            "doc_name": chunk["doc_name"],
            "page_num": int(
                chunk["page_num"]
            ),
            "text": chunk["text"],
        }

        retrieved_results.append(
            retrieved_result
        )

        # 只把来源和文本交给模型。
        # 不加入source_question_ids或其他评测标签。
        evidence_sections.append(
            f"""
[{citation_id}]
Document: {chunk["doc_name"]}
Page: {chunk["page_num"]}
Content:
{chunk["text"]}
""".strip()
        )

    retrieved_evidence_text = "\n\n".join(
        evidence_sections
    )

    system_prompt = build_system_prompt(
        "rag"
    )

    user_prompt = f"""
Company:
{sample["company"]}

Question:
{question}

Retrieved evidence:
{retrieved_evidence_text}
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

    print("正在调用DeepSeek生成RAG回答……")

    # 这里才真正调用DeepSeek API
    raw_result, usage = call_llm_json(
        messages,
        max_tokens=600,
    )

    validated_result = (
        QAAnswer.model_validate(raw_result)
    )

    allowed_citations = {
        result["citation_id"]
        for result in retrieved_results
    }

    invalid_citations = [
        citation
        for citation in validated_result.citations
        if citation not in allowed_citations
    ]

    validation_warnings = []

    if invalid_citations:
        validation_warnings.append(
            "模型使用了不存在的引用编号："
            f"{invalid_citations}"
        )

    if (
        not validated_result.abstain
        and not validated_result.citations
    ):
        validation_warnings.append(
            "模型进行了作答，但没有提供引用。"
        )

    # 以下内容只在模型回答完成后用于检查
    gold_pages = get_gold_pages(sample)

    retrieval_hit = any(
        (
            result["doc_name"],
            result["page_num"],
        )
        in gold_pages
        for result in retrieved_results
    )

    cited_results = [
        result
        for result in retrieved_results
        if result["citation_id"]
        in validated_result.citations
    ]

    cited_gold_page = any(
        (
            result["doc_name"],
            result["page_num"],
        )
        in gold_pages
        for result in cited_results
    )

    if (
        not validated_result.abstain
        and validated_result.citations
        and not cited_gold_page
    ):
        validation_warnings.append(
            "模型进行了作答，但没有引用任何"
            "FinanceBench标准证据页。"
        )

    print("\n" + "=" * 70)
    print("模型回答")
    print("=" * 70)
    print(
        validated_result.model_dump_json(
            indent=2
        )
    )

    print("\n标准答案：")
    print(sample["gold_answer"])

    print("\nToken使用情况：")
    print(usage)

    print("\n" + "=" * 70)
    print(f"Top-{TOP_K}检索结果")
    print("=" * 70)

    for result in retrieved_results:
        page_key = (
            result["doc_name"],
            result["page_num"],
        )

        is_gold_page = (
            page_key in gold_pages
        )

        is_cited = (
            result["citation_id"]
            in validated_result.citations
        )

        print("\n" + "-" * 70)
        print(
            f"引用编号："
            f"{result['citation_id']}"
        )
        print(f"排名：{result['rank']}")
        print(
            f"相似度："
            f"{result['score']:.4f}"
        )
        print(
            f"是否为标准证据页："
            f"{is_gold_page}"
        )
        print(
            f"是否被模型引用："
            f"{is_cited}"
        )
        print(
            f"文档：{result['doc_name']}"
        )
        print(
            f"页码：{result['page_num']}"
        )
        print(
            f"chunk_id："
            f"{result['chunk_id']}"
        )
        print("文本开头：")
        print(result["text"][:500])

    print("\n" + "=" * 70)
    print("单题RAG检查")
    print("=" * 70)
    print(
        f"Top-{TOP_K}是否包含标准证据页："
        f"{retrieval_hit}"
    )
    print(
        f"模型是否引用了标准证据页："
        f"{cited_gold_page}"
    )

    if validation_warnings:
        print("结构、引用编号或离线评测警告：")

        for warning in validation_warnings:
            print(f"- {warning}")
    else:
        print("JSON结构与引用编号合法性检查通过。")
        print(
            "注意：这不代表引用内容一定支持答案，"
            "引用支持性仍需单独评测。"
        )

    print(f"DeepSeek模型：{LLM_MODEL_NAME}")
    print(f"Prompt版本：{PROMPT_VERSION}")
    print(
        "注意：标准答案、gold_pages和"
        "source_question_ids均未发送给模型。"
    )


if __name__ == "__main__":
    main()
