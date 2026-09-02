import hashlib
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evidence_chunk_corpus.jsonl"
)

EMBEDDINGS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evidence_chunk_embeddings.npy"
)

METADATA_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evidence_chunk_embeddings_meta.json"
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


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
    """
    计算所有chunk_id的哈希值。

    它用于保证向量文件中的第i行，
    始终对应语料文件中的第i条记录。
    """

    chunk_id_text = "\n".join(
        record["chunk_id"]
        for record in records
    )

    return hashlib.sha256(
        chunk_id_text.encode("utf-8")
    ).hexdigest()


def inspect_token_lengths(model, texts, records):
    """
    检查文本块的token长度。

    如果存在超过模型限制的文本块，
    就停止程序，避免模型静默截断文本。
    """

    print("\n正在检查文本块Token长度……")

    tokenized = model.tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )

    token_lengths = np.array(
        [
            len(input_ids)
            for input_ids in tokenized["input_ids"]
        ],
        dtype=np.int32,
    )

    max_sequence_length = model.max_seq_length

    over_limit_indices = np.where(
        token_lengths > max_sequence_length
    )[0]

    print("-" * 70)
    print("Token长度统计")
    print("-" * 70)
    print(f"模型最大Token长度：{max_sequence_length}")
    print(f"文本块数量：{len(token_lengths)}")
    print(f"平均Token数：{mean(token_lengths):.2f}")
    print(f"Token数中位数：{median(token_lengths):.2f}")
    print(
        f"90%分位数："
        f"{np.percentile(token_lengths, 90):.2f}"
    )
    print(
        f"95%分位数："
        f"{np.percentile(token_lengths, 95):.2f}"
    )
    print(f"最大Token数：{token_lengths.max()}")
    print(
        f"超过模型限制的文本块："
        f"{len(over_limit_indices)}"
    )

    if len(over_limit_indices) > 0:
        print("\n前5个超长文本块：")

        for index in over_limit_indices[:5]:
            print("-" * 70)
            print(f"chunk_id：{records[index]['chunk_id']}")
            print(f"Token数：{token_lengths[index]}")
            print(
                f"文本开头："
                f"{records[index]['text'][:200]}"
            )

        raise ValueError(
            "发现超过模型最大输入长度的文本块。"
            "程序已停止，没有生成向量。"
            "请先调整文本切块参数。"
        )

    return token_lengths


def main():
    print("=" * 70)
    print("加载文本块语料")
    print("=" * 70)

    records = load_jsonl(INPUT_PATH)

    if not records:
        raise ValueError("文本块语料为空。")

    texts = [
        record["text"]
        for record in records
    ]

    chunk_ids = [
        record["chunk_id"]
        for record in records
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("发现重复的chunk_id。")

    print(f"输入文件：{INPUT_PATH}")
    print(f"文本块数量：{len(records)}")
    print(f"向量模型：{MODEL_NAME}")

    print("\n正在加载向量模型……")

    model = SentenceTransformer(MODEL_NAME)

    print(f"运行设备：{model.device}")
    print(
        f"向量维度："
        f"{model.get_embedding_dimension()}"
    )

    token_lengths = inspect_token_lengths(
        model=model,
        texts=texts,
        records=records,
    )

    print("\n所有文本块均未超过模型长度限制。")
    print("开始生成文本向量……")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"向量维度异常：{embeddings.shape}"
        )

    if embeddings.shape[0] != len(records):
        raise ValueError(
            "向量数量与文本块数量不一致。"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "向量中出现NaN或无穷大。"
        )

    # 因为生成向量时进行了归一化，
    # 所以每个向量的L2范数应接近1
    vector_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    EMBEDDINGS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        EMBEDDINGS_OUTPUT_PATH,
        embeddings,
    )

    metadata = {
        "model_name": MODEL_NAME,
        "source_corpus": str(
            INPUT_PATH.relative_to(PROJECT_ROOT)
        ),
        "chunk_count": len(records),
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "embedding_dtype": str(
            embeddings.dtype
        ),
        "normalized": True,
        "max_sequence_length": int(
            model.max_seq_length
        ),
        "maximum_observed_token_length": int(
            token_lengths.max()
        ),
        "chunk_ids_sha256": calculate_chunk_id_hash(
            records
        ),
    }

    with METADATA_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("文本向量生成完成")
    print("=" * 70)
    print(f"向量形状：{embeddings.shape}")
    print(f"向量数据类型：{embeddings.dtype}")
    print(
        f"向量范数最小值："
        f"{vector_norms.min():.6f}"
    )
    print(
        f"向量范数平均值："
        f"{vector_norms.mean():.6f}"
    )
    print(
        f"向量范数最大值："
        f"{vector_norms.max():.6f}"
    )
    print(
        f"向量保存位置："
        f"{EMBEDDINGS_OUTPUT_PATH}"
    )
    print(
        f"元数据保存位置："
        f"{METADATA_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()