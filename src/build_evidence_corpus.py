import json
from pathlib import Path
from statistics import mean, median

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "external"
    / "financebench"
    / "data"
    / "financebench_open_source.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evidence_page_corpus.jsonl"
)


def main():
    # 读取 FinanceBench 原始数据
    df = pd.read_json(INPUT_PATH, lines=True)

    # 使用“文档名 + 页码”标识一个页面
    page_map = {}

    for _, row in df.iterrows():
        question_id = row["financebench_id"]
        company = row["company"]

        for evidence in row["evidence"]:
            doc_name = evidence.get("doc_name") or row["doc_name"]
            page_num = int(evidence["evidence_page_num"])

            # 严格使用完整证据页，禁止退回局部证据
            full_page_text = evidence.get("evidence_text_full_page")

            if not isinstance(full_page_text, str) or not full_page_text.strip():
                 raise ValueError(
                f"缺少完整证据页面："
                f"question_id={question_id}, "
                f"doc_name={doc_name}, "
                f"page_num={page_num}"
                )

            page_text = full_page_text.strip()

            page_key = (doc_name, page_num)

            if page_key not in page_map:
                page_map[page_key] = {
                    "chunk_id": f"{doc_name}__page_{page_num}",
                    "company": company,
                    "doc_name": doc_name,
                    "page_num": page_num,
                    "text": page_text,

                    # 仅供后续评测使用，不能放进模型提示词
                    "source_question_ids": [question_id],
                }

            else:
                existing = page_map[page_key]

                # 同一页面如果存在多个文本版本，保留较完整的版本
                if len(page_text) > len(existing["text"]):
                    existing["text"] = page_text

                if question_id not in existing["source_question_ids"]:
                    existing["source_question_ids"].append(question_id)

    corpus = list(page_map.values())

    # 固定顺序，保证每次运行结果一致
    corpus.sort(key=lambda x: (x["doc_name"], x["page_num"]))

    for item in corpus:
        item["source_question_ids"].sort()

    chunk_ids = [item["chunk_id"] for item in corpus]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("发现重复的 chunk_id。")

    covered_question_ids = {
        question_id
        for item in corpus
        for question_id in item["source_question_ids"]
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for item in corpus:
            file.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    text_lengths = [len(item["text"]) for item in corpus]
    document_count = len({item["doc_name"] for item in corpus})

    print("=" * 70)
    print("FinanceBench证据页面语料库构建完成")
    print("=" * 70)
    print(f"原始问题数量：{len(df)}")
    print(f"覆盖问题数量：{len(covered_question_ids)}")
    print(f"去重后的文档数量：{document_count}")
    print(f"去重后的页面数量：{len(corpus)}")
    print(f"页面平均字符数：{mean(text_lengths):.2f}")
    print(f"页面字符数中位数：{median(text_lengths):.2f}")
    print(f"页面最短字符数：{min(text_lengths)}")
    print(f"页面最长字符数：{max(text_lengths)}")
    print(f"保存位置：{OUTPUT_PATH}")

    print("\n第一条语料：")
    first_item = corpus[0]
    print(f"chunk_id：{first_item['chunk_id']}")
    print(f"公司：{first_item['company']}")
    print(f"文档：{first_item['doc_name']}")
    print(f"页码：{first_item['page_num']}")
    print(f"关联问题：{first_item['source_question_ids']}")
    print(f"文本开头：{first_item['text'][:300]}")


if __name__ == "__main__":
    main()