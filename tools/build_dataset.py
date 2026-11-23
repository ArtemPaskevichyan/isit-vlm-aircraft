"""Download the approved Commons images, normalize them and build labels.csv.

Outputs:
    data/images/{id}.jpg        1400px wide master copy
    data/images/{id}_model.jpg  max side 1024, q85 - what we send to the model
    data/labels.csv             ground truth + attribution
    tools/manifest.json         hashes for cache keys
"""
import hashlib
import io
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "images"
SEL = json.loads((ROOT / "tools" / "selection.json").read_text())
PLAN = {p["id"]: p for p in json.loads((ROOT / "tools" / "dataset_plan.json").read_text())}
API = "https://commons.wikimedia.org/w/api.php"
UA = "AircraftVLMResearch/1.0 (educational project)"
MODEL_MAX_SIDE = 1024


def get(url: str, params: dict | None = None, timeout: int = 60) -> requests.Response:
    last = None
    for attempt in range(5):
        r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=timeout)
        if r.status_code == 200:
            return r
        last = r
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"HTTP {last.status_code} for {last.url}")


def strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub("<[^>]+>", "", value or "")).strip()


def metadata(title: str) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
    }
    r = get(API, params)
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    ii = page["imageinfo"][0]
    meta = ii.get("extmetadata", {})
    return {
        "source_page": ii.get("descriptionurl"),
        "source_file": ii.get("url"),
        "artist": strip_html(meta.get("Artist", {}).get("value", "")),
        "license": meta.get("LicenseShortName", {}).get("value", ""),
        "license_url": meta.get("LicenseUrl", {}).get("value", ""),
    }


def download(title: str, width: int = 1400) -> Image.Image:
    name = title.split("File:", 1)[-1]
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(name)}?width={width}"
    r = get(url, timeout=120)
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    rows, manifest = [], {}
    for key, title in SEL.items():
        plan = PLAN[key]
        m = metadata(title)
        img = download(title)
        master = IMG / f"{key}.jpg"
        img.save(master, quality=92, optimize=True)

        model_img = img.copy()
        model_img.thumbnail((MODEL_MAX_SIDE, MODEL_MAX_SIDE), Image.LANCZOS)
        model_path = IMG / f"{key}_model.jpg"
        model_img.save(model_path, quality=85, optimize=True)

        sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        manifest[key] = {"sha256": sha, "model_size": list(model_img.size)}
        rows.append(
            {
                "id": key,
                "file": f"images/{key}_model.jpg",
                "master": f"images/{key}.jpg",
                "label": plan["label"],
                "subject": plan["subject"],
                "trap": plan["trap"],
                "why_hard": plan["why_hard"],
                "width": model_img.width,
                "height": model_img.height,
                "sha256": sha,
                "artist": m["artist"],
                "license": m["license"],
                "license_url": m["license_url"],
                "source_page": m["source_page"],
            }
        )
        print(f"{key} {plan['label']:8s} {plan['trap']:2s} {model_img.size} {m['license']}")
        time.sleep(2)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "labels.csv", index=False)
    (ROOT / "tools" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(df)} images -> data/labels.csv")
    print(df["label"].value_counts().to_string())
    print(df["trap"].value_counts().to_string())


if __name__ == "__main__":
    main()
