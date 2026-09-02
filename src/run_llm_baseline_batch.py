"""Run the 30-question LLM-only baseline with qa_protocol_v1.

The script deliberately sends only company and question to the model. Gold
answers and other FinanceBench annotations are written to the result only after
generation so that the raw output remains convenient for later review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_protocol import PROMPT_VERSION, QAAnswer, build_system_prompt
from llm_client import MODEL_NAME, call_llm_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dev_30.jsonl"
EXPECTED_PROMPT_VERSION = "qa_protocol_v1"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw_outputs"
    / "llm_only_dev_qa_protocol_v1.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run the LLM-only baseline on dev_30.jsonl."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path. Existing compatible output is resumed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run at most this many rows (mainly for testing).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum API/validation attempts per question (default: 3).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay in seconds between successful requests (default: 0.3).",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalar values to ordinary JSON values."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {str(key): json_safe(value) for key, value in usage.items()}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {"raw": str(usage)}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append and fsync one record so interruption loses at most the active call."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_completed(
    output_path: Path,
    *,
    expected_model: str,
    expected_prompt_version: str,
    expected_data_hash: str,
) -> set[str]:
    """Load successful IDs and reject accidental mixing of experiment configs."""
    if not output_path.exists():
        return set()

    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Output line {line_number} is not valid JSON. "
                    "Repair or move the file before resuming."
                ) from exc

            checks = {
                "model": expected_model,
                "prompt_version": expected_prompt_version,
                "dev_data_sha256": expected_data_hash,
            }
            for field, expected in checks.items():
                if record.get(field) != expected:
                    raise RuntimeError(
                        f"Existing output has incompatible {field} at line "
                        f"{line_number}: {record.get(field)!r} != {expected!r}. "
                        "Use a different --output path; do not mix experiments."
                    )

            sample_id = str(record.get("sample_id", ""))
            if not sample_id:
                raise RuntimeError(
                    f"Existing output line {line_number} has no sample_id."
                )
            if sample_id in completed:
                raise RuntimeError(
                    f"Existing output contains duplicate sample_id: {sample_id}."
                )
            completed.add(sample_id)
    return completed


def build_record(
    sample: pd.Series,
    result: QAAnswer,
    usage: Any,
    *,
    latency_seconds: float,
    data_hash: str,
) -> dict[str, Any]:
    response = result.model_dump()
    record: dict[str, Any] = {
        "sample_id": str(sample["sample_id"]),
        "question_type": json_safe(sample.get("question_type")),
        "company": json_safe(sample.get("company")),
        "question": json_safe(sample.get("question")),
        "gold_answer": json_safe(sample.get("gold_answer")),
        "justification": json_safe(sample.get("justification")),
        "dataset_issue": json_safe(sample.get("dataset_issue")),
        "predicted_answer": response["answer"],
        "answer": response["answer"],
        "confidence": response["confidence"],
        "abstain": response["abstain"],
        "citations": response["citations"],
        "reason": response["reason"],
        "model": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "mode": "llm_only",
        "max_tokens": 600,
        "dev_data_sha256": data_hash,
        "latency_seconds": round(latency_seconds, 6),
        "usage": usage_to_dict(usage),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return record


def main() -> int:
    args = parse_args()
    output_path = resolve_path(args.output)
    error_path = output_path.with_name(output_path.stem + "_errors.jsonl")

    if PROMPT_VERSION != EXPECTED_PROMPT_VERSION:
        raise RuntimeError(
            f"Expected {EXPECTED_PROMPT_VERSION!r}, got {PROMPT_VERSION!r}. "
            "Change the output name or restore the intended protocol."
        )
    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1.")
    if args.delay < 0:
        raise ValueError("--delay cannot be negative.")
    if not DEV_DATA_PATH.exists():
        raise FileNotFoundError(f"Development set not found: {DEV_DATA_PATH}")

    dev_df = pd.read_json(DEV_DATA_PATH, lines=True)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        dev_df = dev_df.iloc[: args.limit]

    required_columns = {"sample_id", "company", "question", "gold_answer"}
    missing_columns = required_columns - set(dev_df.columns)
    if missing_columns:
        raise RuntimeError(
            "Development set is missing columns: " + ", ".join(sorted(missing_columns))
        )
    if dev_df["sample_id"].astype(str).duplicated().any():
        duplicates = dev_df.loc[
            dev_df["sample_id"].astype(str).duplicated(), "sample_id"
        ].tolist()
        raise RuntimeError(f"Development set has duplicate sample IDs: {duplicates}")

    data_hash = sha256_file(DEV_DATA_PATH)
    completed = load_completed(
        output_path,
        expected_model=MODEL_NAME,
        expected_prompt_version=PROMPT_VERSION,
        expected_data_hash=data_hash,
    )
    system_prompt = build_system_prompt("llm_only")

    print("=" * 72)
    print("LLM Only batch evaluation")
    print("=" * 72)
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Mode: llm_only (no retrieved evidence)")
    print(f"Development set: {DEV_DATA_PATH}")
    print(f"Development rows selected: {len(dev_df)}")
    print(f"Already completed: {len(completed)}")
    print(f"Output: {output_path}")
    print("Gold answers are never included in model messages.")
    print("-" * 72)

    successful_this_run = 0
    failed_ids: list[str] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for row_index, sample in dev_df.iterrows():
        sample_id = str(sample["sample_id"])
        if sample_id in completed:
            print(f"[skip] {sample_id}")
            continue

        user_prompt = (
            f"Company:\n{sample['company']}\n\n"
            f"Question:\n{sample['question']}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                start = time.perf_counter()
                raw_result, usage = call_llm_json(messages, max_tokens=600)
                latency = time.perf_counter() - start
                validated_result = QAAnswer.model_validate(raw_result)
                record = build_record(
                    sample,
                    validated_result,
                    usage,
                    latency_seconds=latency,
                    data_hash=data_hash,
                )
                append_jsonl(output_path, record)

                usage_dict = record["usage"]
                total_prompt_tokens += int(usage_dict.get("prompt_tokens", 0) or 0)
                total_completion_tokens += int(
                    usage_dict.get("completion_tokens", 0) or 0
                )
                successful_this_run += 1
                completed.add(sample_id)
                position = dev_df.index.get_loc(row_index) + 1
                print(
                    f"[ok {position:02d}/{len(dev_df):02d}] {sample_id} | "
                    f"abstain={validated_result.abstain} | "
                    f"confidence={validated_result.confidence} | "
                    f"{latency:.2f}s"
                )
                last_error = None
                break
            except Exception as exc:  # keep the batch alive after API/schema errors
                last_error = exc
                print(
                    f"[retry {attempt}/{args.max_retries}] {sample_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if attempt < args.max_retries:
                    time.sleep(2 ** attempt)

        if last_error is not None:
            failed_ids.append(sample_id)
            append_jsonl(
                error_path,
                {
                    "sample_id": sample_id,
                    "model": MODEL_NAME,
                    "prompt_version": PROMPT_VERSION,
                    "dev_data_sha256": data_hash,
                    "error_type": type(last_error).__name__,
                    "error": str(last_error),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"[failed] {sample_id}; rerun the same command to retry it.")

        if args.delay:
            time.sleep(args.delay)

    expected_ids = set(dev_df["sample_id"].astype(str))
    finished_selected = len(expected_ids & completed)
    print("-" * 72)
    print(f"Completed selected rows: {finished_selected}/{len(expected_ids)}")
    print(f"New successes this run: {successful_this_run}")
    print(f"Failures this run: {len(failed_ids)}")
    print(f"Prompt tokens this run: {total_prompt_tokens}")
    print(f"Completion tokens this run: {total_completion_tokens}")
    print(f"Saved to: {output_path}")
    if failed_ids:
        print("Failed sample IDs: " + ", ".join(failed_ids))
        print(f"Error log: {error_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
