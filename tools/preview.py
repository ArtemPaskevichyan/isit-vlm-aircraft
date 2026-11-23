"""Download preview thumbnails for the selected Commons files and build a contact sheet."""
import io
import json
import math
import sys
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
PREV = ROOT / "tools" / "previews"
SEL = ROOT / "tools" / "selection.json"
UA = "AircraftVLMResearch/1.0 (educational project)"


def fetch(title: str, width: int = 900) -> Image.Image:
    name = title.split("File:", 1)[-1]
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(name)}?width={width}"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=90, allow_redirects=True)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def main():
    sel = json.loads(SEL.read_text())
    PREV.mkdir(parents=True, exist_ok=True)
    thumbs = {}
    for key, title in sel.items():
        try:
            img = fetch(title)
            img.save(PREV / f"{key}.jpg", quality=85)
            thumbs[key] = img
            print(f"{key}: ok {img.size}")
        except Exception as exc:  # noqa: BLE001
            print(f"{key}: FAIL {exc}", file=sys.stderr)

    cols, cell_w, cell_h = 5, 320, 240
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for i, (key, img) in enumerate(thumbs.items()):
        img.thumbnail((cell_w - 10, cell_h - 34))
        x = (i % cols) * cell_w + (cell_w - img.width) // 2
        y = (i // cols) * cell_h + 30
        sheet.paste(img, (x, y))
        draw.text(((i % cols) * cell_w + 8, (i // cols) * cell_h + 4), key, fill="black", font=font)
    out = ROOT / "tools" / "contact_sheet.jpg"
    sheet.save(out, quality=80)
    print("sheet:", out, sheet.size)


if __name__ == "__main__":
    main()
