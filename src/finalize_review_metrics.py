"""Export final reviewed XLSX tables and recompute FinanceBench dev metrics.

Place beside check_final_reviews.py (version 1.1+) in src/, then run:
    python src/finalize_review_metrics.py

Dependencies: openpyxl for read-only XLSX access; otherwise standard library.
No model/API calls. Original XLSX, JSONL, and legacy results are never edited.
Every run creates a NEW directory under results/metrics/final_qa_protocol_v1/.
Only runs containing review_manifest.json completed successfully.

The checker must pass again before export. LLM new_* columns become canonical
column names; legacy comparison columns are excluded from the exported copy.
Human binary labels use 0/1. Inapplicable citation support stays empty in CSV
and becomes null in JSON where appropriate. Text, gold formatting, and notes
are preserved. Clean exclusions are taken from the RAG review and applied to
BOTH systems by sample_id; this shared metadata source is recorded explicitly.

These are descriptive metrics from existing human labels. No truth re-review
or RAG raw-run settings audit is performed. Do not infer a retrieval-only
causal effect until the run-configuration comparison is completed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import check_final_reviews as checker
except ImportError as exc:
    raise SystemExit("请将本脚本与 check_final_reviews.py（1.1版或更新版）放在同一个src文件夹。") from exc


VERSION = "1.0"
HIGH_CONFIDENCE = 0.8
EXCLUSION = "exclude_from_clean_subset_metric"
EXCLUSION_SOURCE = "Dense RAG review: exclude_from_clean_subset_metric, matched by sample_id"
COMMON_RATES = (
    ("accuracy", "correct_count", "sample_count"),
    ("answered_accuracy", "answered_correct_count", "answered_count"),
    ("abstain_rate", "abstain_count", "sample_count"),
    ("source_hallucination_rate_all", "source_hallucination_count", "sample_count"),
    ("source_hallucination_rate_answered", "source_hallucination_answered_count", "answered_count"),
    ("high_confidence_wrong_rate", "high_confidence_wrong_count", "wrong_answered_count"),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def clean_metadata(llm_rows, rag_rows):
    """Reuse explicit exclusions only; never infer exclusions from correctness."""
    metadata = {}
    for row in rag_rows:
        sid = row["sample_id"]
        if EXCLUSION not in row:
            raise ValueError(f"RAG/{sid} 缺少{EXCLUSION}，不能确定Clean口径")
        flag = checker.binary(row[EXCLUSION])
        issue = row.get("dataset_issue")
        if flag and checker.empty(issue):
            raise ValueError(f"RAG/{sid} 已排除但dataset_issue为空，请先说明排除原因")
        metadata[sid] = {EXCLUSION: flag, "dataset_issue": issue}
    if {row["sample_id"] for row in llm_rows} != set(metadata):
        raise ValueError("两组样本ID不一致，不能共享Clean排除名单")
    for row in llm_rows:
        sid = row["sample_id"]
        if EXCLUSION in row and checker.binary(row[EXCLUSION]) != metadata[sid][EXCLUSION]:
            raise ValueError(f"{sid} 两表的Clean排除标记冲突，已停止")
        issue = row.get("dataset_issue")
        if not checker.empty(issue) and not checker.equivalent(issue, metadata[sid]["dataset_issue"]):
            raise ValueError(f"{sid} 两表的dataset_issue不一致，请先核对")
    return metadata


def canonicalize(rows, role, metadata):
    drop = set(checker.ALIASES.values()) | {"answer_exact_match", "abstain_changed", "review_scope"}
    output = []
    for source in rows:
        row = {key: value for key, value in source.items()
               if key not in drop and not key.startswith(("legacy_", "_"))}
        for key in ("manual_correct", "source_hallucination", "abstain"):
            row[key] = checker.binary(row[key])
        row["confidence"] = checker.confidence(row["confidence"])
        row.update(metadata[row["sample_id"]])
        row["clean_exclusion_source"] = EXCLUSION_SOURCE
        if role == "rag":
            row["citation_support_correct"] = checker.binary(row["citation_support_correct"], optional=True)
            if "retrieval_hit_at_5" not in row:
                raise ValueError("RAG表缺少retrieval_hit_at_5，不能计算Hit@5")
            for key in ("retrieval_hit_at_5", "cited_gold_page", "citation_labels_valid"):
                if key in row:
                    row[key] = checker.binary(row[key])
        output.append(row)
    return output


def summarize(rows, role):
    n = len(rows)
    answered = [row for row in rows if not row["abstain"]]
    wrong = [row for row in answered if not row["manual_correct"]]
    result = {
        "sample_count": n,
        "correct_count": sum(row["manual_correct"] for row in rows),
        "answered_count": len(answered),
        "answered_correct_count": sum(row["manual_correct"] for row in answered),
        "abstain_count": n - len(answered),
        "source_hallucination_count": sum(row["source_hallucination"] for row in rows),
        "source_hallucination_answered_count": sum(row["source_hallucination"] for row in answered),
        "wrong_answered_count": len(wrong),
        "high_confidence_wrong_count": sum(row["confidence"] >= HIGH_CONFIDENCE for row in wrong),
        "high_confidence_threshold": HIGH_CONFIDENCE,
    }
    for metric, numerator, denominator in COMMON_RATES:
        result[metric] = rate(result[numerator], result[denominator])
    if role == "rag":
        # The comparison uses answered-only citation support. Abstention claims
        # (if any) must not silently change this denominator in a future run.
        applicable = [row for row in answered if row["citation_support_correct"] is not None]
        if len(applicable) != len(answered):
            raise ValueError("RAG已作答样本存在未审核的引用支持标签")
        result.update({
            "retrieval_hit_at_5_count": sum(row["retrieval_hit_at_5"] for row in rows),
            "citation_support_applicable_count": len(applicable),
            "citation_support_correct_count": sum(row["citation_support_correct"] for row in applicable),
            "citation_support_blank_count": sum(row["citation_support_correct"] is None for row in rows),
            "citation_support_labeled_abstention_count": sum(
                row["abstain"] and row["citation_support_correct"] is not None for row in rows
            ),
        })
        result["retrieval_hit_at_5_rate"] = rate(result["retrieval_hit_at_5_count"], n)
        result["citation_support_accuracy"] = rate(result["citation_support_correct_count"], len(applicable))
    return result


def distribution(rows, key):
    counts = Counter(row[key] for row in rows)
    return [{key: kind, "count": count, "rate_all_samples": rate(count, len(rows))}
            for kind, count in sorted(counts.items())]


def report(rows, role):
    clean = [row for row in rows if not row[EXCLUSION]]
    result = {
        "system": "LLM Only" if role == "llm" else "Dense RAG",
        "status": "descriptive_metrics_from_final_labels; rag_run_settings_not_verified",
        "clean_exclusion_source": EXCLUSION_SOURCE,
        "excluded_sample_ids": sorted(row["sample_id"] for row in rows if row[EXCLUSION]),
        "all": summarize(rows, role),
        "clean": summarize(clean, role),
        "by_type": [],
        "error_types": {},
    }
    for subset, selected in (("all", rows), ("clean", clean)):
        result["error_types"][subset] = distribution(selected, "error_type")
        for kind in sorted({row["question_type"] for row in rows}):
            group = [row for row in selected if row["question_type"] == kind]
            result["by_type"].append({"subset": subset, "question_type": kind, **summarize(group, role)})
    if role == "rag":
        # Includes `none` and correct-but-unreliable answers, denominator is all
        # samples in each subset, NOT merely the number of incorrect answers.
        result["failure_stages"] = {"all": distribution(rows, "failure_stage"),
                                    "clean": distribution(clean, "failure_stage")}
    return result


def comparison(llm_report, rag_report):
    rows = []
    for subset in ("all", "clean"):
        left, right = llm_report[subset], rag_report[subset]
        for metric, numerator, denominator in COMMON_RATES:
            a, b = left[metric], right[metric]
            rows.append({
                "subset": subset, "metric": metric,
                "llm_numerator": left[numerator], "llm_denominator": left[denominator], "llm_rate": a,
                "rag_numerator": right[numerator], "rag_denominator": right[denominator], "rag_rate": b,
                "rag_minus_llm_percentage_points": None if a is None or b is None else (b - a) * 100,
            })
    return rows


def csv_bytes(rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def save_bundle(output_dir, llm_rows, rag_rows, manifest):
    llm_report, rag_report = report(llm_rows, "llm"), report(rag_rows, "rag")
    by_type = [{"system": rep["system"], **row} for rep in (llm_report, rag_report) for row in rep["by_type"]]
    payloads = {
        "llm_only_dev_qa_protocol_v1_manual_review.csv": csv_bytes(llm_rows),
        "dense_rag_dev_manual_review.csv": csv_bytes(rag_rows),
        "llm_only_dev_qa_protocol_v1_metrics.json": json_bytes(llm_report),
        "dense_rag_dev_metrics.json": json_bytes(rag_report),
        "system_comparison_dev.csv": csv_bytes(comparison(llm_report, rag_report)),
        "system_comparison_by_type.csv": csv_bytes(by_type),
    }
    manifest = {**manifest, "status": "complete",
                "artifacts_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}
    payloads["review_manifest.json"] = json_bytes(manifest)
    # Refuse ANY existing output directory, even if empty. Each file is also
    # exclusively created. The completion manifest is always the last write.
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        with (output_dir / name).open("xb") as stream:
            stream.write(payload)
    return llm_report, rag_report


def parse_args():
    here = Path(__file__).resolve().parent
    root = here.parent if here.name == "src" else here
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--llm-review", default="results/metrics/llm_only_dev_qa_protocol_v1_manual_review.xlsx")
    parser.add_argument("--rag-review", default="results/metrics/dense_rag_dev_manual_review.xlsx")
    parser.add_argument("--llm-raw", default="results/raw_outputs/llm_only_dev_qa_protocol_v1.jsonl")
    parser.add_argument("--dev-data", default="data/processed/dev_30.jsonl")
    parser.add_argument("--llm-sheet")
    parser.add_argument("--rag-sheet")
    parser.add_argument("--output-dir", type=Path, help="可选：必须是不存在的新目录，绝不覆盖")
    return parser.parse_args()


def main():
    args = parse_args()
    if not hasattr(checker, "gold_equivalent"):
        raise ValueError("check_final_reviews.py版本过旧，请更新为1.1版")
    root = args.project_root.resolve()
    sources = {key: (root / getattr(args, key)).resolve()
               for key in ("llm_review", "rag_review", "llm_raw", "dev_data")}
    sources["checker_script"] = Path(checker.__file__).resolve()
    sources["export_script"] = Path(__file__).resolve()
    hashes = {key: sha256(path) for key, path in sources.items()}
    started = datetime.now(timezone.utc)
    output_dir = ((root / args.output_dir).resolve() if args.output_dir else
                  root / "results/metrics/final_qa_protocol_v1" / started.strftime("run_%Y%m%dT%H%M%S_%fZ"))
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，不会覆盖：{output_dir}。请去掉--output-dir或指定新目录")

    print("先重新运行只读检查；通过后才导出。不会调用模型API。", flush=True)
    command = [sys.executable, str(sources["checker_script"]), "--project-root", str(root)]
    for key in ("llm_review", "rag_review", "llm_raw", "dev_data"):
        command.extend(["--" + key.replace("_", "-"), str(sources[key])])
    for key in ("llm_sheet", "rag_sheet"):
        if getattr(args, key):
            command.extend(["--" + key.replace("_", "-"), getattr(args, key)])
    subprocess.run(command, check=True)

    llm = checker.read_workbook(sources["llm_review"], "llm", args.llm_sheet)
    rag = checker.read_workbook(sources["rag_review"], "rag", args.rag_sheet)
    metadata = clean_metadata(llm, rag)
    llm, rag = canonicalize(llm, "llm", metadata), canonicalize(rag, "rag", metadata)
    # Detect edits made during validation/reading; never export mixed versions.
    if any(sha256(path) != hashes[key] for key, path in sources.items()):
        raise ValueError("检查期间输入文件发生变化；未导出，请保存Excel后重跑")
    excluded = sorted(sid for sid, row in metadata.items() if row[EXCLUSION])
    manifest = {
        "script_version": VERSION, "created_at_utc": started.isoformat(),
        "sources": {key: {"path": str(path), "sha256": hashes[key]} for key, path in sources.items()},
        "validation": "check_final_reviews.py passed immediately before export",
        "rag_run_settings_verified": False,
        "comparison_status": "descriptive_only_pending_rag_run_configuration_audit",
        "manual_labels_rejudged": False,
        "clean_exclusion_source": EXCLUSION_SOURCE, "excluded_sample_ids": excluded,
        "transformations": [
            "LLM new_* renamed to canonical fields; legacy comparison columns omitted from export only",
            "Binary fields serialized as 0/1; inapplicable citation support left empty",
            "Gold strings, model text, notes and human decisions preserved",
            "RAG dataset_issue and Clean exclusion metadata shared with LLM by sample_id",
        ],
        "metric_definitions": {
            "accuracy": "correct / all samples; abstention is incorrect",
            "answered_accuracy": "correct answered / answered",
            "source_hallucination_rate_all": "source_hallucination=1 / all samples",
            "source_hallucination_rate_answered": "source_hallucination=1 among answered / answered",
            "high_confidence_wrong_rate": "wrong answered with confidence>=0.8 / wrong answered",
            "citation_support_accuracy": "supported answered / answered with citation-support labels (RAG only)",
            "retrieval_hit_at_5_rate": "page-level hits / all samples; does not establish chunk-level support",
            "failure_stage_rates": "stage count / all samples of subset, including none",
            "undefined_rates": "JSON null / CSV empty; never replaced with zero",
            "units": "rates use 0..1; comparison differences use percentage points",
        },
    }
    left, right = save_bundle(output_dir, llm, rag, manifest)
    print("\n导出和统计完成（未修改原表或旧结果）。")
    print("Clean共同排除：" + (", ".join(excluded) if excluded else "无"))
    for rep in (left, right):
        all_rows, clean = rep["all"], rep["clean"]
        print(f"\n{rep['system']}：")
        print(f"  准确率：{all_rows['correct_count']}/{all_rows['sample_count']} = {all_rows['accuracy']:.2%}")
        print(f"  来源幻觉：{all_rows['source_hallucination_count']}/{all_rows['sample_count']} = {all_rows['source_hallucination_rate_all']:.2%}")
        clean_rate = "N/A" if clean['accuracy'] is None else f"{clean['accuracy']:.2%}"
        print(f"  Clean准确率：{clean['correct_count']}/{clean['sample_count']} = {clean_rate}")
    print("\n保存目录：" + str(output_dir))
    print("共7个文件；review_manifest.json记录来源哈希、指标口径和完成状态。")
    print("注意：这是人工标签的描述性比较；RAG运行参数仍待核验，不作因果或显著性结论。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError:
        print("\n检查未通过，未生成任何结果文件。请先处理上面的检查问题。")
        raise SystemExit(1)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"\n停止：{exc}\n原始文件和旧结果未修改。若出现新输出目录，无完成manifest则不可用于报告。")
        raise SystemExit(2)
