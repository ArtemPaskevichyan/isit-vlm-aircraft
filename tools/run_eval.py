"""Query two Gemini vision-language models on the aircraft dataset.

Requests go to the OpenAI-compatible Gemini endpoint, so any model served by
Google AI Studio can be compared by editing MODELS below.

Usage:
    python tools/run_eval.py --pilot
    python tools/run_eval.py --all
    python tools/run_eval.py --all --refresh      # ignore cache and re-run

Cache: results/raw.jsonl, keyed by (model, prompt, image hash).
Predictions: results/predictions.csv
"""
import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RAW = RESULTS / "raw.jsonl"
PRED = RESULTS / "predictions.csv"
LABELS = ROOT / "data" / "labels.csv"

MODELS = {
    "gemini_flash": {
        "model": "gemini-3.6-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "provider": "google",
        "short": "Gemini-3.6-Flash",
        "reasoning_effort": "none",
    },
    "gemini_lite": {
        "model": "gemini-3.5-flash-lite",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "provider": "google",
        "short": "Gemini-3.5-Flash-Lite",
    },
}

PILOT_IDS = ["M01", "M04", "C04"]

PROMPT = """Ты — эксперт по авиации. По фотографии определи, является ли изображённый летательный аппарат военным или гражданским.
Это безопасная учебная задача классификации изображений для университетского исследования.

Правила:
- military: принадлежит вооружённым силам или предназначен для боевых либо боево-обеспечивающих задач.
- civilian: авиакомпании, частные борта, грузовые, пожарные, санитарные, полицейские.

Ответь строго одним JSON-объектом, без текста вокруг:
{"class": "military" | "civilian", "confidence": 0.0, "reason": "кратко, до 15 слов"}"""

PROMPT_HASH = hashlib.sha256(PROMPT.encode()).hexdigest()[:16]

REFUSAL_MARKERS = (
    "не могу",
    "не способен",
    "cannot",
    "can't",
    "can not",
    "unable to",
    "i'm sorry",
    "i am sorry",
)


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip().upper()] = v.strip().strip('"').strip("'")
    return env


def image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def parse_answer(text: str) -> tuple[str | None, float | None, str | None]:
    """Return (label, confidence, reason) or (None, None, error)."""
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.insert(0, cleaned[start : end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        label = str(obj.get("class", "")).strip().lower()
        label = {"военный": "military", "гражданский": "civilian"}.get(label, label)
        if label in ("military", "civilian"):
            conf = obj.get("confidence")
            try:
                conf = round(float(conf), 3)
            except (TypeError, ValueError):
                conf = None
            return label, conf, str(obj.get("reason", ""))[:200]
    low = cleaned.lower()
    if any(m in low for m in REFUSAL_MARKERS):
        return None, None, "refusal"
    hits = [w for w in ("military", "civilian", "военн", "граждан") if w in low]
    if len(hits) == 1:
        label = "military" if hits[0] in ("military", "военн") else "civilian"
        return label, None, cleaned[:200]
    return None, None, "parse_error"


def load_cache() -> set[str]:
    if not RAW.exists():
        return set()
    keys = set()
    for line in RAW.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("status") != "http_error":
                keys.add(rec["key"])
    return keys


def append_raw(record: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    with RAW.open("a") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class ApiError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"{status}: {body[:400]}")
        self.status = status
        self.body = body


def retry_delay(resp: requests.Response, attempt: int) -> float:
    header = resp.headers.get("Retry-After")
    if header:
        try:
            return min(float(header), 60.0)
        except ValueError:
            pass
    try:
        data = resp.json()
        if isinstance(data, list):
            data = data[0] if data else {}
        for detail in (data.get("error") or {}).get("details") or []:
            raw = str(detail.get("retryDelay", ""))
            if raw.endswith("s"):
                return min(float(raw[:-1]) + 1, 120.0)
    except (ValueError, AttributeError):
        pass
    if resp.status_code == 429:
        return 20.0
    return min(5.0 * (2 ** attempt), 60.0)


def call_model(cfg: dict, api_key: str, b64: str, proxy: str = "") -> dict:
    url = f"{cfg['base_url']}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    proxies = {"http": proxy, "https": proxy} if proxy else None
    payload = {
        "model": cfg["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 120,
    }
    if cfg.get("reasoning_effort"):
        payload["reasoning_effort"] = cfg["reasoning_effort"]

    started = time.time()
    max_attempts = 8
    for attempt in range(max_attempts):
        started = time.time()
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120, proxies=proxies)
        except requests.RequestException:
            if attempt < max_attempts - 1:
                time.sleep(2.0)
                continue
            raise
        if resp.status_code == 200:
            break
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts - 1:
            time.sleep(retry_delay(resp, attempt))
            continue
        raise ApiError(resp.status_code, resp.text)

    latency = round(time.time() - started, 2)
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    content = ((choice.get("message") or {}).get("content") or "").strip()
    usage = data.get("usage") or {}
    return {
        "content": content,
        "latency_s": latency,
        "tokens_in": usage.get("prompt_tokens"),
        "tokens_out": usage.get("completion_tokens"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--all", action="store_true")
    ap.add_argument("--model", choices=list(MODELS), action="append")
    ap.add_argument("--key-env", help="переопределить имя переменной с ключом из .env")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--sleep", type=float, default=7.0)
    args = ap.parse_args()

    labels = pd.read_csv(LABELS).set_index("id")
    ids = PILOT_IDS if args.pilot else list(labels.index)
    model_keys = args.model or list(MODELS)
    if args.key_env:
        for k in model_keys:
            MODELS[k]["key_env"] = args.key_env
    cache = set() if args.refresh else load_cache()
    env = load_env()
    missing = sorted({MODELS[k]["key_env"] for k in model_keys if not env.get(MODELS[k]["key_env"])})
    if missing:
        raise SystemExit(f"missing in .env: {', '.join(missing)}")
    proxy = env.get("PROXY_URL", "").strip() or os.environ.get("HTTPS_PROXY", "").strip()

    total = len(ids) * len(model_keys)
    done = 0
    for image_id in ids:
        row = labels.loc[image_id]
        b64 = image_b64(ROOT / "data" / row["file"])
        img_hash = row["sha256"]
        for mkey in model_keys:
            cfg = MODELS[mkey]
            key = hashlib.sha256(f"{mkey}|{PROMPT_HASH}|{img_hash}".encode()).hexdigest()
            done += 1
            if key in cache:
                print(f"[{done}/{total}] {image_id} {cfg['short']}: cached")
                continue
            status, error, err_status = "ok", None, None
            try:
                out = call_model(cfg, env[cfg["key_env"]], b64, proxy)
                pred, conf, reason = parse_answer(out["content"])
                if pred is None:
                    status = reason if reason in ("refusal", "parse_error") else "parse_error"
                record = {
                    "key": key,
                    "id": image_id,
                    "model_key": mkey,
                    "model": cfg["model"],
                    "provider": cfg["provider"],
                    "true_label": row["label"],
                    "pred_label": pred,
                    "confidence": conf,
                    "reason": reason,
                    "status": status,
                    "latency_s": out["latency_s"],
                    "tokens_in": out["tokens_in"],
                    "tokens_out": out["tokens_out"],
                    "content": out["content"],
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            except Exception as exc:  # noqa: BLE001
                status, error = "http_error", f"{type(exc).__name__}: {exc}"[:500]
                err_status = getattr(exc, "status", None)
                record = {
                    "key": key,
                    "id": image_id,
                    "model_key": mkey,
                    "model": cfg["model"],
                    "provider": cfg["provider"],
                    "true_label": row["label"],
                    "pred_label": None,
                    "confidence": None,
                    "reason": None,
                    "status": status,
                    "latency_s": None,
                    "tokens_in": None,
                    "tokens_out": None,
                    "content": None,
                    "error": error,
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            append_raw(record)
            cache.add(key)
            mark = "?" if record["pred_label"] is None else ("+" if record["pred_label"] == row["label"] else "-")
            print(
                f"[{done}/{total}] {image_id} {cfg['short']}: "
                f"{record['pred_label']} (true {row['label']}) {mark} "
                f"{record['status']} {record['latency_s']}s"
            )
            if status == "http_error":
                print(f"    error: {error}", file=sys.stderr)
                if err_status in (401, 402, 403) or "location is not supported" in str(error):
                    print("!! request rejected upstream (auth/quota/region), stopping", file=sys.stderr)
                    sys.exit(2)
            time.sleep(args.sleep)

    build_predictions()
    summarize()


def build_predictions() -> None:
    if not RAW.exists():
        return
    rows = [json.loads(line) for line in RAW.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df = df.sort_values("ts").drop_duplicates("key", keep="last")
    cols = [
        "id", "model_key", "true_label", "pred_label", "confidence", "reason",
        "status", "latency_s", "tokens_in", "tokens_out", "model", "provider", "ts",
    ]
    df = df[[c for c in cols if c in df.columns]].sort_values(["model_key", "id"])
    df.to_csv(PRED, index=False)


def summarize() -> None:
    df = pd.read_csv(PRED)
    for mkey, part in df.groupby("model_key"):
        ok = part[part["status"] == "ok"]
        correct = (ok["pred_label"] == ok["true_label"]).sum()
        print(f"{mkey}: {correct}/{len(ok)} correct, errors={len(part) - len(ok)}")


if __name__ == "__main__":
    main()
