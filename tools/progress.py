"""Show collection progress from results/raw.jsonl.

Usage:
    python tools/progress.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw.jsonl"


def main() -> None:
    if not RAW.exists():
        print("no results yet")
        return
    rows = [json.loads(line) for line in RAW.read_text().splitlines() if line.strip()]
    labels = [line.split(",")[0] for line in (ROOT / "data" / "labels.csv").read_text().splitlines()[1:]]
    got: dict[str, set[str]] = {}
    errors = 0
    for rec in rows:
        if rec.get("status") == "http_error":
            errors += 1
        else:
            got.setdefault(rec["id"], set()).add(rec["model_key"])

    total = len(labels) * 2
    done = sum(len(v) for v in got.values())
    print(f"requests ok: {done}/{total}   http errors (not cached): {errors}")
    print(f"images with both models: {sum(1 for v in got.values() if len(v) == 2)}/{len(labels)}")
    missing = [i for i in labels if len(got.get(i, set())) < 2]
    if missing:
        print("missing:")
        for i in missing:
            print(f"  {i}: {', '.join(sorted({'gemini_flash', 'gemini_lite'} - got.get(i, set())))}")


if __name__ == "__main__":
    main()
