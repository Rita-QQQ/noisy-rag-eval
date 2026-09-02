#!/usr/bin/env python3
"""Offline, read-only validation and descriptive metrics for the confirmed noise run.

Python 3.9+; standard library only. Run from the project root. No API calls.
--run accepts the original run directory or its ZIP. Inputs are never modified.
Every invocation creates a new output directory; manifest.json is written last.
The workbook supplies accepted labels; this script does not rejudge answers.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import posixpath
import re
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

VERSION = "noise_metrics_v1.0"
LEVELS = (0.0, 0.2, 0.4, 0.6)
CLEAN_EXCLUDED = {"financebench_id_00283"}
DEFAULT_RUN = "cross_company_replace_v1_20260901T143232Z_62caa52d"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
HEADERS = "sample_id noise_fraction status predicted_answer manual_correct citation_support_correct source_hallucination error_type failure_stage notes review_complete gold_answer reason confidence abstain citations company question question_type attempt_count clean_excluded gold_page_present prompt_tokens_all_attempts completion_tokens_all_attempts generation_seconds case_id request_sha256 source_reference annotation_origin gold_justification".split()
RULES = {
    "end_to_end_accuracy": "correct / all planned cases; schema failures score 0",
    "valid_output_accuracy": "correct / schema-valid outputs",
    "answered_accuracy": "correct non-abstentions / valid non-abstentions",
    "abstain_rate_all": "valid abstentions / planned cases; failures are not abstentions",
    "abstain_rate_valid": "valid abstentions / schema-valid outputs",
    "citation_support_accuracy_applicable": "support=1 / nonblank support labels; includes substantive abstentions",
    "citation_support_accuracy_answered": "support=1 among valid non-abstentions / valid non-abstentions",
    "source_hallucination_rate_valid": "hallucination=1 / schema-valid outputs; failures have no label",
    "source_hallucination_rate_answered": "hallucination=1 among valid non-abstentions / valid non-abstentions",
    "source_hallucination_incidence_all": "flagged cases / planned cases; incidence only, NOT an assertion that failed cases are hallucination-free",
    "high_confidence_error_rate_wrong_answers": "wrong non-abstentions with confidence>=0.8 / wrong valid non-abstentions",
    "source_hallucination_definition": "invented or falsely attributed source content; insufficient support, calculation errors and disclosed assumptions alone do not qualify",
    "clean": "exclude only financebench_id_00283 in every condition; retain all other disputes",
    "pairing": "same sample_id paired against this run's 0% condition; no historical run substituted",
    "unknown_denominator": "undefined ratios are JSON null / blank CSV, never zero",
    "limitations": "30 development questions, one seed (2026); replacement removes evidence and inserts distractor candidates; descriptive, not a significance or pure-distraction causal claim",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_workbook(data):
    """Read XLSX values directly; never depend on cached metric formulas."""
    out = {}
    with ZipFile(io.BytesIO(data)) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(t.text or "" for t in si.findall(".//s:t", NS))
                       for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        rel = {e.get("Id"): e.get("Target") for e in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        sheets = ET.fromstring(z.read("xl/workbook.xml")).findall("s:sheets/s:sheet", NS)
        for sheet in sheets:
            rid = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            target = rel[rid]
            name = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            cells = {}
            for c in ET.fromstring(z.read(name)).findall("s:sheetData/s:row/s:c", NS):
                kind, node = c.get("t"), c.find("s:v", NS)
                raw = node.text if node is not None else None
                if kind == "s":
                    value = strings[int(raw)]
                elif kind == "inlineStr":
                    value = "".join(t.text or "" for t in c.findall("s:is//s:t", NS))
                elif raw is None:
                    value = None
                elif kind in ("str", "e"):
                    value = raw
                else:
                    value = float(raw)
                    if value.is_integer():
                        value = int(value)
                require(kind != "e", "Excel error at %s!%s: %s" % (sheet.get("name"), c.get("r"), value))
                cells[c.get("r")] = {"value": value, "formula": c.find("s:f", NS) is not None}
            out[sheet.get("name")] = cells
    require("Review" in out and "Readme" in out, "Missing Review or Readme sheet")
    return out


def col_number(text):
    result = 0
    for ch in text:
        result = result * 26 + ord(ch) - ord("A") + 1
    return result


def read_review(book):
    matrix = defaultdict(dict)
    for ref, c in book["Review"].items():
        match = re.fullmatch(r"([A-Z]+)([0-9]+)", ref)
        matrix[int(match[2])][col_number(match[1])] = c
    header = matrix[4]
    names = [header.get(i, {}).get("value") for i in range(1, len(HEADERS) + 1)]
    require(names == HEADERS, "Review row 4 headers differ from the documented schema")
    rows = []
    for number, cells in sorted(matrix.items()):
        if number < 5 or not any(c.get("value") is not None for c in cells.values()):
            continue
        require(not any(c["formula"] for c in cells.values()), "Review row %s contains a formula in raw fields" % number)
        row = {name: cells.get(i, {}).get("value") for i, name in enumerate(names, 1)}
        require(row["sample_id"] and row["case_id"], "Incomplete Review row %s" % number)
        rows.append(row)
    return rows


def validate_labels(rows):
    require(len(rows) == 120, "Expected exactly 120 review rows")
    require(len({r["case_id"] for r in rows}) == 120, "Duplicate case_id")
    groups = defaultdict(list)
    for r in rows:
        cid = r["case_id"]
        require(r["noise_fraction"] in LEVELS, cid + ": invalid noise fraction")
        count = LEVELS.index(r["noise_fraction"])
        require(cid == "%s__replace_%s_of_5" % (r["sample_id"], count), cid + ": ID/fraction mismatch")
        groups[r["sample_id"]].append(r["noise_fraction"])
        require(r["review_complete"] == 1, cid + ": confirmation not recorded")
        for field in ("manual_correct", "clean_excluded", "gold_page_present"):
            require(r[field] in (0, 1), cid + ": invalid " + field)
        require(r["clean_excluded"] == int(r["sample_id"] in CLEAN_EXCLUDED), cid + ": Clean exclusion changed")
        require(r["status"] in ("succeeded", "failed"), cid + ": unknown status")
        require(r["failure_stage"] in {"none", "retrieval", "generation", "retrieval_and_generation", "dataset", "unclear", "protocol"}, cid + ": invalid failure_stage")
        require(r["error_type"] in {"correct", "wrong_numeric", "wrong_conclusion", "incomplete", "abstention", "dataset_issue", "schema_failure"}, cid + ": invalid error_type")
        if r["status"] == "failed":
            require(r["manual_correct"] == 0 and r["error_type"] == "schema_failure" and r["failure_stage"] == "protocol", cid + ": invalid failure labels")
            require(r["annotation_origin"] == "rule_prefill_checked", cid + ": invalid failure check state")
            require(all(r[k] is None for k in ("citation_support_correct", "source_hallucination", "predicted_answer", "reason", "confidence", "abstain", "citations")), cid + ": schema failure must not become a valid answer or zero-valued N/A")
            continue
        require(r["annotation_origin"] == "reviewed_confirmed", cid + ": stale review state")
        require(r["source_hallucination"] in (0, 1), cid + ": missing hallucination label")
        require(r["abstain"] in (0, 1), cid + ": invalid abstain")
        require(isinstance(r["confidence"], (int, float)) and 0 <= r["confidence"] <= 1, cid + ": invalid confidence")
        require(r["citation_support_correct"] in (0, 1, None), cid + ": invalid support label")
        if r["abstain"]:
            require(r["manual_correct"] == 0 and r["error_type"] == "abstention", cid + ": abstention scoring mismatch")
        else:
            require(r["citation_support_correct"] is not None, cid + ": missing applicable support label")
            require(r["error_type"] not in ("abstention", "schema_failure"), cid + ": wrong output category")
        require((r["manual_correct"] == 1) == (r["error_type"] == "correct"), cid + ": correctness/category mismatch")
        if r["failure_stage"] == "none":
            require(r["manual_correct"] == 1 and r["citation_support_correct"] == 1 and r["source_hallucination"] == 0, cid + ": inconsistent failure_stage=none")
    require(len(groups) == 30 and all(sorted(v) == list(LEVELS) for v in groups.values()), "Need the same 30 questions at all four levels")
    require(CLEAN_EXCLUDED <= set(groups), "Missing predeclared Clean exclusion")


def read_run(path):
    filenames = ("results.jsonl", "run_summary.json", "run_manifest.json")
    if path.is_dir():
        data = {name: (path / name).read_bytes() for name in filenames}
        archive_sha = None
    else:
        archive = path.read_bytes()
        archive_sha = sha(archive)
        with ZipFile(io.BytesIO(archive)) as z:
            matches = [n for n in z.namelist() if n == "results.jsonl" or n.endswith("/results.jsonl")]
            require(len(matches) == 1, "ZIP must contain exactly one run/results.jsonl")
            prefix = matches[0][:-len("results.jsonl")]
            data = {name: z.read(prefix + name) for name in filenames}
    raw = [json.loads(line) for line in data["results.jsonl"].decode("utf-8-sig").splitlines() if line.strip()]
    return raw, json.loads(data["run_summary.json"]), json.loads(data["run_manifest.json"]), {name: sha(b) for name, b in data.items()}, archive_sha


def validate_source(rows, raw, summary, manifest, book, archive_sha):
    require(summary.get("complete") is True and summary.get("sources_unchanged_at_end") is True, "Run incomplete or sources changed")
    require(summary.get("not_attempted_case_ids") == [] and summary.get("planned_cases") == 120, "Not all planned cases were attempted")
    require(summary["run_id"] == manifest["run_id"] == book["Readme"]["B22"]["value"], "Run ID mismatch")
    if archive_sha:
        require(archive_sha == book["Readme"]["B23"]["value"], "ZIP differs from the archive recorded in Readme!B23")
    require(len(raw) == len(rows) and len({r["case_id"] for r in raw}) == len(rows), "Raw cases missing or duplicated")
    by_id = {r["case_id"]: r for r in raw}
    require(set(by_id) == {r["case_id"] for r in rows}, "Raw/review case IDs do not match")
    for r in rows:
        cid, source = r["case_id"], by_id[r["case_id"]]
        require(source["run_id"] == summary["run_id"], cid + ": wrong source run")
        response = source["model_response"] or {}
        expected = {k: source[k] for k in ("sample_id", "company", "question", "question_type", "status", "attempt_count", "request_sha256", "gold_answer")}
        expected.update({"gold_justification": source["justification"], "predicted_answer": response.get("answer"), "reason": response.get("reason"), "confidence": response.get("confidence"), "abstain": int(response["abstain"]) if response else None, "citations": ", ".join(response.get("citations", [])) or None,
                         "noise_fraction": source["noise_audit"]["replacement_fraction_of_blocks"], "clean_excluded": int(source["noise_audit"]["exclude_from_clean_subset_metric"]), "gold_page_present": int(source["noise_audit"]["context_gold_page_present"])})
        for key, value in expected.items():
            require(r[key] == value, cid + ": raw field changed: " + key)
        require(len(source["evidence_context"]) == 5, cid + ": expected five evidence blocks")
        require(len(source["attempts"]) == source["attempt_count"], cid + ": attempts do not reconcile")
        usage = source["usage_all_attempts"]
        for token in ("prompt", "completion", "total"):
            require(usage.get(token + "_tokens_complete") is True, cid + ": unknown token usage; cannot treat as zero")
            vals = [(a.get("usage") or {}).get(token + "_tokens") for a in source["attempts"]]
            require(all(isinstance(v, int) and v >= 0 for v in vals), cid + ": missing per-attempt token usage")
            require(sum(vals) == usage[token + "_tokens_known_sum"], cid + ": per-attempt usage mismatch")
        for token in ("prompt", "completion"):
            require(r[token + "_tokens_all_attempts"] == usage[token + "_tokens_known_sum"], cid + ": workbook token sum mismatch")
        require(math.isclose(r["generation_seconds"], source["timing"]["generation_total_seconds"], rel_tol=1e-12, abs_tol=1e-9), cid + ": timing mismatch")
    for state in ("succeeded", "failed"):
        require(summary[state] == sum(r["status"] == state for r in rows), "Summary status count mismatch")
    totals = summary["usage_all_attempts"]
    require(totals["attempt_count"] == sum(r["attempt_count"] for r in rows), "Attempt total mismatch")
    for token in ("prompt", "completion"):
        require(totals[token + "_tokens_complete"] is True and totals[token + "_tokens_known_sum"] == sum(r[token + "_tokens_all_attempts"] for r in rows), "Run token total mismatch")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(rows):
    valid = [r for r in rows if r["status"] == "succeeded"]
    answered = [r for r in valid if not r["abstain"]]
    applicable = [r for r in valid if r["citation_support_correct"] is not None]
    wrong = [r for r in answered if not r["manual_correct"]]
    n, v, a = len(rows), len(valid), len(answered)
    correct = sum(r["manual_correct"] for r in rows)
    support = sum(r["citation_support_correct"] for r in applicable)
    hallucinations = sum(r["source_hallucination"] for r in valid)
    support_answered = sum(r["citation_support_correct"] for r in answered)
    hallucinations_answered = sum(r["source_hallucination"] for r in answered)
    high_errors = sum(r["confidence"] >= 0.8 for r in wrong)
    return {
        "planned_count": n, "valid_output_count": v, "schema_failure_count": n-v,
        "answered_count": a, "abstain_count": v-a, "correct_count": correct,
        "end_to_end_accuracy": ratio(correct, n), "valid_output_accuracy": ratio(correct, v),
        "answered_accuracy": ratio(sum(r["manual_correct"] for r in answered), a),
        "schema_failure_rate": ratio(n-v, n), "abstain_rate_all": ratio(v-a, n), "abstain_rate_valid": ratio(v-a, v),
        "citation_applicable_count": len(applicable), "citation_supported_count": support,
        "citation_support_accuracy_applicable": ratio(support, len(applicable)),
        "citation_supported_answered_count": support_answered, "citation_support_accuracy_answered": ratio(support_answered, a),
        "source_hallucination_count": hallucinations, "source_hallucination_rate_valid": ratio(hallucinations, v),
        "source_hallucination_answered_count": hallucinations_answered, "source_hallucination_rate_answered": ratio(hallucinations_answered, a),
        "source_hallucination_incidence_all": ratio(hallucinations, n),
        "wrong_answered_count": len(wrong), "high_confidence_error_count": high_errors,
        "high_confidence_error_rate_wrong_answers": ratio(high_errors, len(wrong)),
        "gold_page_present_count": sum(r["gold_page_present"] for r in rows),
        "attempt_count": sum(r["attempt_count"] for r in rows),
        "prompt_tokens_all_attempts": sum(r["prompt_tokens_all_attempts"] for r in rows),
        "completion_tokens_all_attempts": sum(r["completion_tokens_all_attempts"] for r in rows),
        "mean_generation_seconds_all_cases": ratio(sum(r["generation_seconds"] for r in rows), n),
    }


def calculate(rows):
    by_level, by_type, distributions, paired = [], [], [], []
    for subset in ("all", "clean"):
        selected = [r for r in rows if subset == "all" or not r["clean_excluded"]]
        baseline = {r["sample_id"]: r for r in selected if r["noise_fraction"] == 0}
        for level in LEVELS:
            group = [r for r in selected if r["noise_fraction"] == level]
            base = {"subset": subset, "noise_fraction": level}
            by_level.append({**base, **metrics(group)})
            for kind in sorted({r["question_type"] for r in group}):
                by_type.append({**base, "question_type": kind, **metrics([r for r in group if r["question_type"] == kind])})
            for field in ("error_type", "failure_stage"):
                for label, count in sorted(Counter(r[field] for r in group).items()):
                    distributions.append({**base, "field": field, "label": label, "count": count, "denominator": len(group), "rate": count/len(group)})
            if level:
                transitions = Counter((baseline[r["sample_id"]]["manual_correct"], r["manual_correct"]) for r in group)
                paired.append({**base, "paired_count": len(group), "both_correct": transitions[1, 1], "correct_to_incorrect": transitions[1, 0], "incorrect_to_correct": transitions[0, 1], "both_incorrect": transitions[0, 0], "accuracy_change_percentage_points": 100 * (transitions[0, 1] - transitions[1, 0]) / len(group)})
    return {"by_level": by_level, "by_type": by_type, "distributions": distributions, "paired_vs_zero": paired}


def write_csv(path, rows, fields=None):
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def export(output_root, rows, result, provenance):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = output_root / ("run_" + stamp + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    write_csv(directory / "noise_review_labels.csv", rows, HEADERS)
    for name in ("by_level", "by_type", "distributions", "paired_vs_zero"):
        write_csv(directory / ("noise_" + name + ".csv"), result[name])
    write_json(directory / "noise_metrics.json", {"script_version": VERSION, "definitions": RULES, **result})
    artifacts = {p.name: sha(p.read_bytes()) for p in sorted(directory.iterdir())}
    write_json(directory / "manifest.json", {"complete": True, "script_version": VERSION, "generated_at_utc": stamp, "sources": provenance, "clean_excluded_sample_ids": sorted(CLEAN_EXCLUDED), "definitions": RULES, "output_sha256": artifacts,
               "verification_scope": "Confirmed labels, all 120 Review source records against results.jsonl, per-case attempts and usage against run summary. Workbook display excerpts are not byte-exact source inputs. Does not independently reproduce retrieval, rejudge labels, or infer reviewer identity."})
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=Path("results/metrics/noise_formal_manual_review.xlsx"))
    parser.add_argument("--run", type=Path, default=Path("results/noise_runs") / DEFAULT_RUN, help="Original formal-run directory or ZIP")
    parser.add_argument("--output-root", type=Path, default=Path("results/metrics/noise_formal_final"))
    parser.add_argument("--check-only", action="store_true", help="Validate and print, without creating any files")
    args = parser.parse_args(argv)
    try:
        data = args.review.read_bytes()
        book = read_workbook(data)
        rows = read_review(book)
        validate_labels(rows)
        raw, summary, manifest, source_hashes, archive_sha = read_run(args.run)
        validate_source(rows, raw, summary, manifest, book, archive_sha)
        result = calculate(rows)
        print("Validation passed: 120 confirmed rows; same 30 questions at four levels.")
        print("Schema failures are retained; N/A labels are not converted to zero.")
        print("Level   Correct/all   Clean          Abstain/all   Failed/all   Support/applicable")
        for i, level in enumerate(LEVELS):
            m, c = result["by_level"][i], result["by_level"][i+4]
            print("%3.0f%%    %2d/%d=%5.2f%%  %2d/%d=%5.2f%%  %2d/%d         %d/%d         %d/%d" % (level*100, m["correct_count"], m["planned_count"], 100*m["end_to_end_accuracy"], c["correct_count"], c["planned_count"], 100*c["end_to_end_accuracy"], m["abstain_count"], m["planned_count"], m["schema_failure_count"], m["planned_count"], m["citation_supported_count"], m["citation_applicable_count"]))
        require(sha(args.review.read_bytes()) == sha(data), "Workbook changed during calculation")
        _, _, _, hashes_now, archive_now = read_run(args.run)
        require(hashes_now == source_hashes and archive_now == archive_sha, "Run sources changed during calculation")
        if args.check_only:
            print("CHECK_ONLY: no files created; no API calls.")
            return 0
        provenance = {"review_path": str(args.review.resolve()), "review_sha256": sha(data), "run_path": str(args.run.resolve()), "run_id": summary["run_id"], "run_file_sha256": source_hashes, "run_zip_sha256": archive_sha, "script_sha256": sha(Path(__file__).read_bytes()), "model": manifest["requested_model"], "prompt_version": manifest["prompt_version"], "generation": manifest["generation"], "review_confirmation_count": sum(r["review_complete"] == 1 for r in rows)}
        output = export(args.output_root, rows, result, provenance)
        print("Saved 7 files to: " + str(output.resolve()))
        print("No API calls; input workbook, formal run and old results were not modified.")
        print("Descriptive development-set results only; no significance or causal claims.")
        return 0
    except (ValueError, KeyError, OSError, ET.ParseError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
