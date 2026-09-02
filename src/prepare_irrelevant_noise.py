"""Offline, frozen-context cross-company replacement experiment.

Default: build a new plan, or verify it read-only if it already exists.
--check-only: verify an existing plan without writing anything.
No model imports, tokenizers, network, credentials, or API calls.
Gold diagnostics and noise masks live ONLY in audit files, not model inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

VERSION = "cross_company_replace_v1"
SEED = 2026
LEVELS = (0, 1, 2, 3)
K = 5
MAX_LENGTH_RELATIVE_DELTA = 0.10
SOURCE_FILES = {
    "dev": "data/processed/dev_30.jsonl",
    "corpus": "data/processed/evidence_chunk_corpus.jsonl",
    "rag": "results/raw_outputs/dense_rag_dev.jsonl",
}
# Freeze the exact user-approved inputs; changing data requires a new plan.
EXPECTED_SOURCE_HASHES = {
    "dev": "fd10fe0e7a62c61b1178153d95e16ff7b5eb4de3dcda8098cf8fbc4f9ed45377",
    "corpus": "eeb86450d787dba09e5e769594607034e4fd99faff0775e850eb089ce8915634",
    "rag": "c54bf1eb99d4d37d53c10706ea772806157f42e003dd35d4b76b8aecc5b3fed8",
}
MODEL_INPUT_FIELDS = {"company", "question", "evidence"}
EVIDENCE_FIELDS = {"citation_id", "doc_name", "page_num", "text"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def norm(value):
    return " ".join(str(value).casefold().split())


def seeded_hash(*parts):
    return digest([SEED, *parts])


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        require(isinstance(row, dict), f"{path.name}:{number} 不是JSON对象")
        rows.append(row)
    return rows


def index_unique(rows, key):
    result = {}
    for row in rows:
        value = row.get(key)
        require(isinstance(value, str) and value and value not in result, f"{key}缺失或重复：{value!r}")
        result[value] = row
    return result


def page_key(doc, page):
    try:
        number = Decimal(str(page))
    except InvalidOperation as exc:
        raise ValueError("页码不是合法数字") from exc
    require(number.is_finite() and number == number.to_integral_value(), "页码必须是整数")
    return norm(doc), int(number)


def gold_pages(question):
    return {page_key(e.get("doc_name") or question["doc_name"], e["evidence_page_num"])
            for e in question["evidence"]}


def evidence_block(chunk, slot):
    return {"citation_id": f"E{slot + 1}", "doc_name": chunk["doc_name"],
            "page_num": chunk["page_num"], "text": chunk["text"]}


def validate_sources(dev, corpus, raw):
    require(len(dev) == len(raw) == 30 and len(corpus) == 615, "应为30题、30条原始检索和615个块")
    dm, cm, rm = index_unique(dev, "sample_id"), index_unique(corpus, "chunk_id"), index_unique(raw, "sample_id")
    require(set(dm) == set(rm), "开发集与原始检索的ID不一致")
    require(dict(Counter(q["question_type"] for q in dev)) ==
            {"domain-relevant": 10, "metrics-generated": 10, "novel-generated": 10}, "问题类型数量不符")
    for chunk in corpus:
        require(all(isinstance(chunk.get(k), str) and chunk[k].strip() for k in ("company", "doc_name", "text")), "语料关键文本为空")
        require(type(chunk.get("content_token_count")) is int and chunk["content_token_count"] > 0, "块Token计数非法")
        require(isinstance(chunk.get("source_question_ids"), list), "语料缺少来源问题ID列表")
        page_key(chunk["doc_name"], chunk["page_num"])
    for q in dev:
        sid = q["sample_id"]
        r = rm[sid]
        for key in ("company", "question", "question_type", "gold_answer"):
            require(r[key] == q[key], f"{sid}原始检索与开发集的{key}不同")
        require(r["system"] == "dense_rag" and r["retriever"] == "dense" and r["top_k"] == 5,
                f"{sid}不是既定Dense Top-5")
        require(r["prompt_version"] == "qa_protocol_v1", f"{sid}提示词版本异常")
        require(type(r["exclude_from_clean_subset_metric"]) is bool, f"{sid}Clean标记必须为布尔值")
        require(not r["exclude_from_clean_subset_metric"] or r.get("dataset_issue"), f"{sid}排除原因缺失")
        found = r["retrieved_results"]
        require(len(found) == 5 and len({e["chunk_id"] for e in found}) == 5, f"{sid}Top-5不足或重复")
        pages = gold_pages(q)
        for slot, block in enumerate(found):
            require(block["rank"] == slot + 1 and block["citation_id"] == f"E{slot+1}", f"{sid}原始排名/引用错位")
            require(block["chunk_id"] in cm, f"{sid}原始块不在语料中")
            chunk = cm[block["chunk_id"]]
            for key in ("company", "doc_name", "text", "page_num"):
                require(block[key] == chunk[key], f"{sid}/{block['chunk_id']}的{key}与语料不一致")
            require(block["is_gold_page"] == (page_key(block["doc_name"], block["page_num"]) in pages), f"{sid}原始gold页标记不符")
        require(r["retrieval_hit_at_5"] == any(e["is_gold_page"] for e in found), f"{sid}Hit@5标记不符")
    return dm, cm, rm


def candidate_pool(question, original, corpus):
    sid = question["sample_id"]
    blocked_ids = {c["chunk_id"] for c in original}
    blocked_docs = {norm(c["doc_name"]) for c in original} | {norm(question["doc_name"])}
    blocked_docs |= {norm(e.get("doc_name") or question["doc_name"]) for e in question["evidence"]}
    blocked_texts = {norm(c["text"]) for c in original}
    company_mention = re.compile(r"(?<!\w)" + re.escape(question["company"]) + r"(?!\w)", re.IGNORECASE)
    return [c for c in corpus if norm(c["company"]) != norm(question["company"])
            and c["chunk_id"] not in blocked_ids and norm(c["doc_name"]) not in blocked_docs
            and sid not in c["source_question_ids"] and not company_mention.search(c["text"])
            and norm(c["text"]) not in blocked_texts]


def choose_replacements(question, original, corpus):
    sid = question["sample_id"]
    slots = sorted(range(K), key=lambda i: (seeded_hash(sid, "slot", i), i))[:max(LEVELS)]
    pool = candidate_pool(question, original, corpus)
    chosen = {}
    docs, companies, texts = set(), set(), set()
    for slot in slots:
        length = original[slot]["content_token_count"]
        eligible = [c for c in pool if norm(c["doc_name"]) not in docs and norm(c["company"]) not in companies
                    and norm(c["text"]) not in texts
                    and abs(c["content_token_count"] - length) / length <= MAX_LENGTH_RELATIVE_DELTA]
        require(eligible, f"{sid}/E{slot+1}没有满足长度要求的候选；不会静默放宽规则")
        replacement = min(eligible, key=lambda c: (abs(c["content_token_count"] - length),
                           seeded_hash(sid, "candidate", slot, c["chunk_id"]), c["chunk_id"]))
        chosen[slot] = replacement
        docs.add(norm(replacement["doc_name"]))
        companies.add(norm(replacement["company"]))
        texts.add(norm(replacement["text"]))
    return slots, chosen, len(pool)


def case_id(sid, count):
    return f"{sid}__replace_{count}_of_5"


def construct(dev, corpus, raw):
    _, cm, rm = validate_sources(dev, corpus, raw)
    inputs, audits, assignments = [], [], []
    for q in dev:
        sid = q["sample_id"]
        r = rm[sid]
        original = [cm[e["chunk_id"]] for e in r["retrieved_results"]]
        slots, chosen, pool_size = choose_replacements(q, original, corpus)
        pages = gold_pages(q)
        base_tokens = sum(c["content_token_count"] for c in original)
        for order, slot in enumerate(slots, 1):
            a, b = original[slot], chosen[slot]
            assignments.append({"sample_id": sid, "slot": slot+1, "first_active_replacement_count": order,
                "question": q["question"], "target_company": q["company"],
                "original_chunk_id": a["chunk_id"], "replacement_chunk_id": b["chunk_id"],
                "replacement_company": b["company"], "replacement_doc": b["doc_name"],
                "replacement_page": b["page_num"], "original_content_tokens": a["content_token_count"],
                "replacement_content_tokens": b["content_token_count"],
                "relative_length_delta": (b["content_token_count"]-a["content_token_count"])/a["content_token_count"],
                "replacement_text": b["text"], "manual_irrelevance_confirmed": None})
        for count in LEVELS:
            selected_slots = set(slots[:count])
            selected = [chosen[i] if i in selected_slots else c for i,c in enumerate(original)]
            model_input = {"company": q["company"], "question": q["question"],
                           "evidence": [evidence_block(c,i) for i,c in enumerate(selected)]}
            identifier = case_id(sid,count)
            inputs.append({"case_id": identifier, "sample_id": sid, "model_input": model_input,
                           "model_input_sha256": digest(model_input)})
            gold_mask = [page_key(c["doc_name"],c["page_num"]) in pages for c in selected]
            current_tokens = sum(c["content_token_count"] for c in selected)
            audits.append({"case_id": identifier, "sample_id": sid, "question_type": q["question_type"],
                "replacement_count": count, "replacement_fraction_of_blocks": count/K,
                "baseline_chunk_ids": [c["chunk_id"] for c in original],
                "context_chunk_ids": [c["chunk_id"] for c in selected],
                "replacement_slot_order": [i+1 for i in slots],
                "replaced_slots": sorted(i+1 for i in selected_slots), "noise_mask": [i in selected_slots for i in range(K)],
                "candidate_pool_size": pool_size, "reference_token_count_baseline": base_tokens,
                "reference_token_count_context": current_tokens,
                "reference_token_relative_delta": (current_tokens-base_tokens)/base_tokens,
                "baseline_gold_page_present": r["retrieval_hit_at_5"],
                "context_gold_page_present": any(gold_mask), "context_gold_page_mask": gold_mask,
                "exclude_from_clean_subset_metric": r["exclude_from_clean_subset_metric"],
                "dataset_issue": r.get("dataset_issue"), "semantic_irrelevance_status": "heuristic_candidates_not_human_certified"})
    # Rotate condition order across shuffled questions. Each within-question
    # position receives each level either 7 or 8 times; do not run all 0% first.
    questions_order = sorted(dev,key=lambda q:(seeded_hash("execution_question",q["sample_id"]),q["sample_id"]))
    levels_order = sorted(LEVELS,key=lambda k:(seeded_hash("execution_level",k),k))
    execution = []
    for n,q in enumerate(questions_order):
        shift = n % len(LEVELS)
        execution.extend(case_id(q["sample_id"],k) for k in levels_order[shift:] + levels_order[:shift])
    validate_plan(inputs,audits,assignments,execution,dev,corpus,raw)
    return inputs,audits,assignments,execution


def validate_plan(inputs,audits,assignments,execution,dev,corpus,raw):
    dm,cm,rm = validate_sources(dev,corpus,raw)
    im,am = index_unique(inputs,"case_id"),index_unique(audits,"case_id")
    require(len(inputs)==len(audits)==120 and len(assignments)==90,"应为120组输入、120组审计和90次唯一替换安排")
    require(set(im)==set(am)==set(execution) and len(execution)==120,"执行顺序缺失或重复case")
    for sid,q in dm.items():
        base = [cm[e["chunk_id"]] for e in rm[sid]["retrieved_results"]]
        slots,chosen,pool_size = choose_replacements(q,base,corpus)
        previous = set()
        for count in LEVELS:
            identifier=case_id(sid,count);entry=im[identifier];audit=am[identifier];mi=entry["model_input"]
            require(entry['sample_id']==audit['sample_id']==sid,f"{identifier}sample_id错位")
            require(set(mi)==MODEL_INPUT_FIELDS and len(mi["evidence"])==5,f"{identifier}模型输入包含非法字段或非5块")
            require(mi["company"]==q["company"] and mi["question"]==q["question"],f"{identifier}题目改变")
            require(entry["model_input_sha256"]==digest(mi),f"{identifier}输入哈希不符")
            active=set(slots[:count]);actual={i for i,(a,b) in enumerate(zip(audit["context_chunk_ids"],audit["baseline_chunk_ids"])) if a!=b}
            require(actual==active and previous<=active,f"{identifier}替换数量/嵌套关系错误")
            previous=active
            require(audit["replacement_count"]==count and audit["replacement_fraction_of_blocks"]==count/5,f"{identifier}噪声比例错误")
            require(audit["candidate_pool_size"]==pool_size,f"{identifier}候选池数量错误")
            require(audit['baseline_chunk_ids']==[c['chunk_id'] for c in base],f"{identifier}基础块ID改变")
            require(audit['replaced_slots']==sorted(i+1 for i in active) and audit['noise_mask']==[i in active for i in range(K)],f"{identifier}噪声掩码不符")
            selected_chunks=[chosen[i] if i in active else base[i] for i in range(K)]
            page_mask=[page_key(c['doc_name'],c['page_num']) in gold_pages(q) for c in selected_chunks]
            require(audit['context_gold_page_mask']==page_mask and audit['context_gold_page_present']==any(page_mask),f"{identifier}页级诊断不符")
            require(audit['reference_token_count_context']==sum(c['content_token_count'] for c in selected_chunks),f"{identifier}长度汇总不符")
            require(audit['exclude_from_clean_subset_metric']==rm[sid]['exclude_from_clean_subset_metric'],f"{identifier}Clean标记改变")
            for i,block in enumerate(mi["evidence"]):
                selected=chosen[i] if i in active else base[i]
                require(set(block)==EVIDENCE_FIELDS and block==evidence_block(selected,i),f"{identifier}/E{i+1}文本、来源或引用编号改变")
                require(audit["context_chunk_ids"][i]==selected["chunk_id"],f"{identifier}审计块ID不符")
    require(len({(a['sample_id'],a['slot']) for a in assignments})==90,"替换安排重复")


def summary_stats(audits,assignments):
    return {
        "sample_count":30,"case_count":120,"blocks_per_case":5,"seed":SEED,
        "replacement_assignments":len(assignments),
        "distinct_replacement_chunks":len({r["replacement_chunk_id"] for r in assignments}),
        "exact_reference_length_matches":sum(r["original_content_tokens"]==r["replacement_content_tokens"] for r in assignments),
        "max_replacement_reference_length_delta":max(abs(r["relative_length_delta"]) for r in assignments),
        "max_context_reference_length_delta":max(abs(r["reference_token_relative_delta"]) for r in audits),
        "excluded_sample_ids":sorted({r["sample_id"] for r in audits if r["exclude_from_clean_subset_metric"]}),
        "by_level":[{"replacement_count":k,"block_fraction":k/5,"cases":sum(r["replacement_count"]==k for r in audits),
                     "gold_page_present_count":sum(r["context_gold_page_present"] for r in audits if r["replacement_count"]==k)} for k in LEVELS],
    }


def preview_markdown(dev,assignments,stats):
    lines=["# 无关块替换方案预览（离线）", "", "这是跨公司替换候选，不是已经人工确认的全部语义无关证据。", "",
        "固定原始Top-5，每题替换0/1/2/3块；比例按块数计算，不是按模型Token计算。",
        "不保护原始gold页：本实验同时包含证据移除和无关块加入，不能单独归因于干扰。", "",
        f"共120个case、90次唯一替换安排；其中{stats['exact_reference_length_matches']}次使用相同语料Token计数。",
        "长度计数来自现有语料，不代表DeepSeek Token严格相同。Gold页命中不等于答案证据充分。", "",
        "## 六题抽样预览", "", "每种题型按sample_id取前两题，仅作快速阅读；全部替换文本见replacement_candidates.jsonl。", ""]
    selected=[]
    for kind in sorted({q["question_type"] for q in dev}):
        selected.extend(sorted([q for q in dev if q["question_type"]==kind],key=lambda q:q["sample_id"])[:2])
    for q in selected:
        lines.extend([f"### {q['sample_id']} / {q['company']} / {q['question_type']}","",q["question"],""])
        for row in assignments:
            if row["sample_id"]!=q["sample_id"]:continue
            lines.extend([f"替换E{row['slot']}（从{row['first_active_replacement_count']}/5档起生效）：{row['replacement_company']}，{row['replacement_doc']}，page_num={row['replacement_page']}（保留语料记录值，不换算印刷页码）。",
                          f"语料Token计数：{row['original_content_tokens']} → {row['replacement_content_tokens']}。",""])
            lines.extend("> "+line for line in row["replacement_text"].splitlines())
            lines.append("")
    return "\n".join(lines)+"\n"


def jsonl_bytes(rows):
    return ("\n".join(canonical(row) for row in rows)+"\n").encode("utf-8")


def prepare_payloads(dev,corpus,raw,hashes):
    inputs,audits,assignments,execution=construct(dev,corpus,raw)
    stats=summary_stats(audits,assignments)
    payloads={"noise_inputs.jsonl":jsonl_bytes(inputs),"noise_audit.jsonl":jsonl_bytes(audits),
              "replacement_candidates.jsonl":jsonl_bytes(assignments),
              "execution_order.json":(json.dumps(execution,indent=2)+"\n").encode(),
              "noise_preview.md":preview_markdown(dev,assignments,stats).encode("utf-8")}
    manifest={"plan_version":VERSION,"status":"offline_generated_heuristic_candidates",
        "api_calls_made_by_builder":0,"seed":SEED,"top_k":K,"replacement_counts":list(LEVELS),
        "sources":{k:{"project_relative_path":SOURCE_FILES[k],"sha256":hashes[k]} for k in SOURCE_FILES},
        "builder_sha256":file_hash(Path(__file__).resolve()),"summary":stats,
        "selection_rules":["Other company; target name absent from candidate text (literal boundary check)",
            "Exclude original Top-5 IDs/texts/documents and target/gold documents",
            "Exclude chunks annotated as sources for the current sample",
            "Distinct replacement companies, documents and normalized texts within each question",
            "Nearest available content_token_count within 10%; SHA256 seeded tie-breaking",
            "Sample-specific seeded replacement positions independent of model answers and scores",
            "Nested replacements and fixed E1..E5 positions across conditions"],
        "execution_order_policy":"seeded question order; rotated condition order balanced to 7/8 appearances per position",
        "limitations":["Cross-company irrelevance is a heuristic, not human certification",
            "0% means zero injected replacements, not perfectly relevant retrieval or a Clean scoring subset",
            "Replacement can remove needed evidence; this is not an evidence-preserving pure-distraction test",
            "Reference token matching uses corpus metadata, not the target model's exact tokenization",
            "Corpus is the supplied benchmark collection; no claim of full-report enterprise retrieval realism",
            "One fixed corruption seed; no uncertainty over multiple corruption realizations estimated",
            "Baseline/noise model calls must be run anew in the same experiment; do not reuse legacy 40% as control"],
        "prompt_boundary":"Send only model_input.company, question and evidence via the shared RAG prompt renderer; never audit labels or condition names",
        "artifacts_sha256":{name:hashlib.sha256(value).hexdigest() for name,value in payloads.items()}}
    payloads["noise_manifest.json"]=(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    return payloads,stats


def verify_existing(folder,payloads):
    for name,expected in payloads.items():
        path=folder/name
        require(path.is_file(),f"缺少{name}；目录可能只写了一部分。不会自动覆盖，请先检查")
        require(path.read_bytes()==expected,f"{name}与可复现方案不一致。不会覆盖；不要直接手工改输入或重新抽样")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir",type=Path,default=Path("data/noise_experiments/cross_company_replace_v1"))
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    root=args.project_root.resolve();output=(root/args.output_dir).resolve()
    paths={key:root/name for key,name in SOURCE_FILES.items()}
    hashes={key:file_hash(path) for key,path in paths.items()}
    for key,value in hashes.items():
        require(value==EXPECTED_SOURCE_HASHES[key],f"{paths[key].name}内容哈希不同于已审核版本；停止，不更改文件")
    dev,corpus,raw=(read_jsonl(paths[key]) for key in ("dev","corpus","rag"))
    payloads,stats=prepare_payloads(dev,corpus,raw,hashes)
    require(all(file_hash(path)==hashes[key] for key,path in paths.items()),"准备期间源文件变化；停止")
    if output.exists():
        verify_existing(output,payloads)
        print("已有方案校验通过；没有修改任何输出文件。")
    else:
        require(not args.check_only,"方案目录尚不存在；先不带--check-only生成")
        output.mkdir(parents=True,exist_ok=False)
        # Completion manifest is written last; all files use exclusive creation.
        for name,data in payloads.items():
            with (output/name).open("xb") as stream:stream.write(data)
        verify_existing(output,payloads)
        print("新方案生成并校验通过；原始数据及旧实验结果未修改。")
    print(f"开发集：30题；语料：615块；原始Top-5：150次块记录均匹配。")
    print("随机种子：2026；0%/20%/40%/60%各30组，共120份输入，每份5块。")
    print(f"90次替换安排中，{stats['exact_reference_length_matches']}次语料Token计数完全相同。")
    print(f"单块最大长度偏差：{stats['max_replacement_reference_length_delta']:.2%}；整组最大偏差：{stats['max_context_reference_length_delta']:.2%}。")
    print("Gold页出现数（仅页级诊断，不等于证据充分）：")
    for row in stats['by_level']:print(f"  {row['block_fraction']:.0%}: {row['gold_page_present_count']}/30")
    print("输出目录："+str(output))
    print("未调用API；未生成模型答案。请先审阅noise_preview.md，不要直接开始120次调用。")
    return 0


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(f"停止：{exc}\n原始文件和旧实验结果未修改。若有不完整新目录，请先检查，不要覆盖。")
        raise SystemExit(2)
