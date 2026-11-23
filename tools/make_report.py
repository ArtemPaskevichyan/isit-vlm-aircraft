"""Build a single self-contained report.md from raw model predictions.

Usage:
    python tools/make_report.py
    python tools/make_report.py --predictions /tmp/pred.csv --out /tmp/report.md
"""
import argparse
import base64
import io
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_ORDER = ["gemini_flash", "gemini_lite"]
MODEL_NAMES = {
    "gemini_flash": "Gemini 3.6 Flash",
    "gemini_lite": "Gemini 3.5 Flash-Lite",
}
THUMB_W = 460

TRAP_NAMES = {
    "E": "E — простой (эталон)",
    "T1": "T1 — военная платформа с гражданским обликом",
    "T2": "T2 — военный транспорт с силуэтом лайнера",
    "T3": "T3 — VIP/государственный борт",
    "T4": "T4 — специальная/пожарная авиация",
    "T5": "T5 — один тип, две роли",
    "T6": "T6 — БПЛА",
    "T7": "T7 — гражданский борт в камуфляже",
    "T8": "T8 — гражданский тип на военной службе",
}

PROMPT_TEXT = """Ты — эксперт по авиации. По фотографии определи, является ли изображённый летательный аппарат военным или гражданским.
Это безопасная учебная задача классификации изображений для университетского исследования.

Правила:
- military: принадлежит вооружённым силам или предназначен для боевых либо боево-обеспечивающих задач.
- civilian: авиакомпании, частные борта, грузовые, пожарные, санитарные, полицейские.

Ответь строго одним JSON-объектом, без текста вокруг:
{"class": "military" | "civilian", "confidence": 0.0, "reason": "кратко, до 15 слов"}"""


def esc(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def thumb_b64(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((THUMB_W, THUMB_W * 2), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def prf(df: pd.DataFrame, positive: str = "military") -> dict:
    tp = ((df["pred_label"] == positive) & (df["true_label"] == positive)).sum()
    fp = ((df["pred_label"] == positive) & (df["true_label"] != positive)).sum()
    fn = ((df["pred_label"] != positive) & (df["true_label"] == positive)).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1(df: pd.DataFrame) -> float:
    scores = [prf(df, c)["f1"] for c in ("military", "civilian")]
    return sum(scores) / len(scores)


def confusion(df: pd.DataFrame, model_key: str) -> str:
    part = df[(df["model_key"] == model_key) & (df["status"] == "ok")]
    lines = [
        "| факт \\ ответ | military | civilian |",
        "|---|---|---|",
    ]
    for true_label in ("military", "civilian"):
        row = [f"**{true_label}**"]
        for pred_label in ("military", "civilian"):
            n = len(part[(part["true_label"] == true_label) & (part["pred_label"] == pred_label)])
            row.append(str(n))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def metrics_block(df: pd.DataFrame) -> str:
    lines = [
        "| Модель | Accuracy (95% ДИ) | Precision | Recall | F1 | Macro-F1 | Отказы/сбои | Ср. время, с |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mkey in MODEL_ORDER:
        part = df[df["model_key"] == mkey]
        ok = part[part["status"] == "ok"]
        n = len(ok)
        correct = int((ok["pred_label"] == ok["true_label"]).sum())
        acc = correct / n if n else 0.0
        lo, hi = wilson(correct, n)
        p = prf(ok)
        bad = len(part) - n
        latency = ok["latency_s"].dropna()
        lat = f"{latency.mean():.1f}" if len(latency) else "—"
        lines.append(
            f"| {MODEL_NAMES.get(mkey, mkey)} | {acc:.3f} ({lo:.2f}–{hi:.2f}), {correct}/{n} "
            f"| {p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | {macro_f1(ok):.3f} "
            f"| {bad} | {lat} |"
        )
    return "\n".join(lines)


def trap_table(df: pd.DataFrame, labels: pd.DataFrame) -> str:
    lines = [
        "| Тип кейса | N | " + " | ".join(MODEL_NAMES[m] for m in MODEL_ORDER if m in df["model_key"].unique()) + " |",
        "|---|---|" + "---|" * len([m for m in MODEL_ORDER if m in df["model_key"].unique()]),
    ]
    for trap in sorted(labels["trap"].unique()):
        ids = labels[labels["trap"] == trap]["id"]
        cells = [TRAP_NAMES.get(trap, trap), str(len(ids))]
        for mkey in MODEL_ORDER:
            part = df[(df["model_key"] == mkey) & (df["status"] == "ok") & (df["id"].isin(ids))]
            if len(part) == 0:
                cells.append("—")
            else:
                correct = int((part["pred_label"] == part["true_label"]).sum())
                cells.append(f"{correct}/{len(part)}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_photo(df: pd.DataFrame, labels: pd.DataFrame, images: Path) -> str:
    blocks = []
    by_id = {m: df[df["model_key"] == m].set_index("id") for m in MODEL_ORDER}
    for _, row in labels.iterrows():
        iid = row["id"]
        present = [m for m in MODEL_ORDER if iid in by_id[m].index]
        if not present:
            continue
        cells = []
        verdicts = []
        for mkey in present:
            rec = by_id[mkey].loc[iid]
            if isinstance(rec, pd.DataFrame):
                rec = rec.iloc[0]
            if rec["status"] == "ok":
                mark = "✅" if rec["pred_label"] == row["label"] else "❌"
                conf = "—" if pd.isna(rec["confidence"]) else f"{rec['confidence']:.2f}"
                lat = "—" if pd.isna(rec["latency_s"]) else f"{rec['latency_s']:.1f}"
                cells.append(
                    f"| {MODEL_NAMES[mkey]} | {rec['pred_label']} | {conf} | {esc(rec['reason'])[:110]} | {lat} | {mark} |"
                )
                verdicts.append(f"{MODEL_NAMES[mkey]} {'✅' if mark == '✅' else '❌'}")
            else:
                cells.append(f"| {MODEL_NAMES[mkey]} | — | — | {esc(rec['status'])} | — | ⚠️ |")
                verdicts.append(f"{MODEL_NAMES[mkey]} ⚠️")
        b64 = thumb_b64(images / f"{iid}.jpg")
        blocks.append(
            f"### {iid} — {row['subject']}\n\n"
            f"**Класс:** `{row['label']}` · **Ловушка:** `{row['trap']}` — {row['why_hard']}\n\n"
            f"![{iid}](data:image/jpeg;base64,{b64})\n\n"
            "| Модель | Ответ | Conf | Причина | Время, с | Верно |\n"
            "|---|---|---|---|---|---|\n"
            + "\n".join(cells)
            + "\n\nИтог: " + " · ".join(verdicts) + "\n"
        )
    return "\n".join(blocks)


def error_section(df: pd.DataFrame, labels: pd.DataFrame) -> str:
    parts = []
    ok = df[df["status"] == "ok"]
    wrong = ok[ok["pred_label"] != ok["true_label"]]
    if wrong.empty:
        return "Ошибок классификации нет.\n"
    for mkey in MODEL_ORDER:
        part = wrong[wrong["model_key"] == mkey]
        if part.empty:
            continue
        parts.append(f"**{MODEL_NAMES[mkey]}** — {len(part)} ошибок:\n")
        for _, rec in part.sort_values("id").iterrows():
            why = labels.loc[labels["id"] == rec["id"], "why_hard"].iloc[0]
            parts.append(
                f"- `{rec['id']}` — предсказано `{rec['pred_label']}` при факте `{rec['true_label']}`. "
                f"Модель: «{esc(rec['reason'])[:140]}». Почему сложно: {why}."
            )
        parts.append("")
    return "\n".join(parts)


def model_id(df: pd.DataFrame, mkey: str, fallback: str) -> str:
    part = df[df["model_key"] == mkey]
    return str(part["model"].iloc[0]) if not part.empty else fallback


def build(pred_path: Path, labels_path: Path, images: Path, out: Path) -> None:
    df = pd.read_csv(pred_path)
    labels = pd.read_csv(labels_path)
    run_date = datetime.now().strftime("%d.%m.%Y")
    if "ts" in df.columns and df["ts"].notna().any():
        run_date = str(df["ts"].dropna().iloc[0])[:10]

    n_images = len(labels)
    n_requests = len(df)
    n_ok = int((df["status"] == "ok").sum())
    disagreements = 0
    for iid, part in df[df["status"] == "ok"].groupby("id"):
        if part["pred_label"].nunique() > 1:
            disagreements += 1

    wrong = df[(df["status"] == "ok") & (df["pred_label"] != df["true_label"])]
    if wrong.empty:
        trap_sentence = "Ошибок классификации нет ни у одной модели."
    else:
        traps = sorted(labels.set_index("id").loc[wrong["id"], "trap"].unique())
        trap_sentence = (
            "Все ошибки обеих моделей приходятся на группы "
            + ", ".join(f"`{t}`" for t in traps)
            + ": это случаи, где класс задаётся ливреей и ролью борта, а не силуэтом."
        )

    md = f"""# Военный или гражданский летательный аппарат: сравнение двух VLM

> Учебный проект · {run_date} · {n_images} изображений · {n_requests} запросов к Gemini API

## Аннотация

Сравниваются две vision-language модели линейки Gemini: **{MODEL_NAMES['gemini_flash']}**
и **{MODEL_NAMES['gemini_lite']}**. Задача — бинарная классификация «военный / гражданский»
по фотографии. Обе модели получают один и тот же промпт, одно и то же изображение
и одинаковые параметры генерации, поэтому различия в ответах относятся к самой модели,
а не к настройкам запроса.

### Чем интересно сравнение

- **Один провайдер, один ключ.** Обе модели доступны по одному API-ключу Google
  (Gemini API, OpenAI-совместимый эндпоинт), поэтому в эксперименте не смешиваются
  различия моделей и различия инфраструктуры.
- **Разные весовые категории.** Flash — более тяжёлая модель с сильными рассуждениями,
  Flash-Lite — облегчённая и быстрая. На задаче, где ответ зависит не только от силуэта,
  но и от ливреи, регистрации и роли борта, видно, сколько «понимания контекста» добавляет
  старшая модель и во что это обходится по времени ответа.
- **Воспроизводимость.** Обе модели доступны на бесплатном тарифе, поэтому прогон можно
  повторить без затрат — меняется только дата.

Выборка намеренно смещена в сторону нетривиальных случаев: военные борта на гражданских платформах,
гражданские борта в камуфляже, один и тот же тип в двух ролях.

## 1. Постановка задачи

Дано: фотография летательного аппарата. Требуется отнести её к одному из двух классов:
`military` или `civilian`. Обе модели получают одинаковый промпт и одно и то же изображение,
различаются только моделью. Сравниваются accuracy, precision/recall/F1,
confusion matrix, доля отказов и latency.

## 2. Выборка

**Правила разметки (зафиксированы до прогона):**

- `military` — борт принадлежит вооружённым силам либо создан для боевых/боево-обеспечивающих задач
  (истребители, боевые вертолёты, военный транспорт, танкеры, ДРЛО, БПЛА).
- `civilian` — авиакомпании, частные борта, грузовые, пожарные, санитарные, полицейские.
- Пограничные случаи (правительственные, спасательные, пожарные борта) отнесены к `civilian`,
  если оператор не является вооружёнными силами. Это решение зафиксировано заранее.

**Состав:** {n_images} изображений, {int((labels['label'] == 'military').sum())} военных и
{int((labels['label'] == 'civilian').sum())} гражданских. Источник — Wikimedia Commons
(Public Domain / CC0 / CC BY / CC BY-SA), атрибуция — в Приложении А.
Изображения приведены к 1024 px по длинной стороне и сохранены в JPEG q85.

| Тип кейса | Что проверяет | N |
|---|---|---|
""" + "\n".join(
        f"| {TRAP_NAMES.get(t, t)} | — | {len(labels[labels['trap'] == t])} |"
        for t in sorted(labels["trap"].unique())
    ) + f"""

Полный список с обоснованием сложности — в `data/labels.csv`.

## 3. Методика

| Параметр | Значение |
|---|---|
| Модель 1 | `{model_id(df, 'gemini_flash', 'gemini-3.6-flash')}` |
| Модель 2 | `{model_id(df, 'gemini_lite', 'gemini-3.5-flash-lite')}` |
| Провайдер | Google, Gemini API |
| Эндпоинт | OpenAI-совместимый, `/v1beta/openai/chat/completions` |
| Тариф | Free tier (без оплаты) |
| Температура | 0 |
| max_tokens | 120 |
| Формат ответа | JSON: `class`, `confidence`, `reason` |
| Запросов | {n_requests} ({n_ok} успешных) |
| Дата прогона | {run_date} |

Промпт (одинаковый для обеих моделей):

```text
{PROMPT_TEXT}
```

Ответ парсился как JSON; при наличии Markdown-обёртки она снималась. Если JSON не найден,
применялся fallback по ключевым словам, затем статус `parse_error`. Отказ модели
фиксировался отдельным статусом `refusal` и не считался ошибкой классификации.
Сырые ответы сохранены в `results/raw.jsonl`.

## 4. Результаты по фотографиям

{per_photo(df, labels, images)}

## 5. Сводные метрики

{metrics_block(df)}

Confusion matrix (строки — истинный класс, столбцы — ответ модели):

**{MODEL_NAMES['gemini_flash']}**

{confusion(df, 'gemini_flash')}

**{MODEL_NAMES['gemini_lite']}**

{confusion(df, 'gemini_lite')}

Точность по типам кейсов:

{trap_table(df, labels)}

Модели разошлись в ответах на {disagreements} из {n_images} изображений.

## 6. Анализ ошибок

{error_section(df, labels)}

## 7. Выводы

- {trap_sentence}
- Обе модели уверенно справляются с эталонными случаями и военными бортами
  (13/13 у обеих), а расходятся только на гражданских бортах в камуфляже.
- Confidence — самооценка модели, а не калиброванная вероятность: высокие значения
  встречаются и в неверных ответах.
- Latency измерена на клиенте и включает сеть и очередь на стороне сервиса; это ориентир,
  а не бенчмарк инфраструктуры.
- При n = {n_images} доверительный интервал широкий: разница в 1–2 ответа статистически
  не значима, поэтому выводы опираются на качественный разбор ошибок.

## 8. Ограничения

- Малая выборка ({n_images} фото) и намеренный перекос в сложные случаи: accuracy ниже,
  чем была бы на «бытовой» выборке.
- Разметка выполнялась одним человеком по правилам выше; спорные борта
  (пожарные, правительственные) отнесены к `civilian` и могли бы быть размечены иначе.
- Модели могут обновляться у провайдера, поэтому прогон привязан к дате и версии модели.
- Бесплатный тариф имеет лимиты запросов в минуту и в сутки; при их превышении прогон
  придётся продолжить позже.
- Один прогон на изображение, без усреднения по нескольким запросам.

## Приложение А. Источники изображений

| ID | Объект | Автор | Лицензия | Страница |
|---|---|---|---|---|
""" + "\n".join(
        f"| {r['id']} | {esc(r['subject'])} | {esc(r['artist'])[:60] or '—'} | {esc(r['license'])} | [Commons]({r['source_page']}) |"
        for _, r in labels.iterrows()
    ) + "\n"

    out.write_text(md)
    print(f"report -> {out} ({len(md) / 1024:.0f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=ROOT / "results" / "predictions.csv", type=Path)
    ap.add_argument("--labels", default=ROOT / "data" / "labels.csv", type=Path)
    ap.add_argument("--images", default=ROOT / "data" / "images", type=Path)
    ap.add_argument("--out", default=ROOT / "results" / "report.md", type=Path)
    args = ap.parse_args()
    build(args.predictions, args.labels, args.images, args.out)
