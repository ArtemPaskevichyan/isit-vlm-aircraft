"""Search Wikimedia Commons for candidate photos for the dataset.

Usage:
    python tools/commons_search.py
Writes tools/candidates.json and prints a compact shortlist per item.
"""
import json
import re
import sys
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
UA = "AircraftVLMResearch/1.0 (educational project; contact: student@example.com)"
ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "tools" / "dataset_plan.json"
OUT = ROOT / "tools" / "candidates.json"

BAD = re.compile(
    r"cockpit|cabin|engine|diagram|map\b|logo|patch|insignia|roundel|drawing|"
    r"painting|scale model|toy|simulator|1:(72|48|32|144|200|400|100)\b|museum interior|"
    r"cargo area|interior|maintenance|mechanics|nose art|noseart|graffiti|"
    r"crash|wreck|wreckage|damaged",
    re.I,
)
GOOD = re.compile(r"in flight|flying|landing|takeoff|take-off|airshow|runway|approach", re.I)


def search(query: str, limit: int = 25):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1600,
    }
    r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=40)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    out = []
    for p in pages.values():
        ii = (p.get("imageinfo") or [{}])[0]
        title = p.get("title", "")
        if ii.get("mime") not in ("image/jpeg", "image/png"):
            continue
        w, h = ii.get("width", 0), ii.get("height", 0)
        if w < 1200 or h < 700:
            continue
        if not (1.05 <= w / h <= 4.0):
            continue
        if BAD.search(title):
            continue
        meta = ii.get("extmetadata", {})
        out.append(
            {
                "title": title,
                "width": w,
                "height": h,
                "mime": ii.get("mime"),
                "size_mb": round(ii.get("size", 0) / 1e6, 2),
                "thumb_url": ii.get("thumburl"),
                "file_url": ii.get("url"),
                "description_url": ii.get("descriptionurl"),
                "artist": re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip(),
                "license": meta.get("LicenseShortName", {}).get("value", ""),
            }
        )
    out.sort(key=lambda x: (0 if GOOD.search(x["title"]) else 1, -x["width"]))
    return out


def main():
    plan = json.loads(PLAN.read_text())
    all_candidates = {}
    for item in plan:
        try:
            cands = search(item["query"])
        except Exception as exc:  # noqa: BLE001
            print(f"!! {item['id']} search failed: {exc}", file=sys.stderr)
            cands = []
        all_candidates[item["id"]] = cands
        print(f"\n=== {item['id']} {item['subject']} ({item['label']}, {item['trap']})")
        for i, c in enumerate(cands[:7], 1):
            print(f"  {i}. {c['title']}  [{c['width']}x{c['height']}, {c['size_mb']} MB]")
    OUT.write_text(json.dumps(all_candidates, ensure_ascii=False, indent=1))
    print(f"\nSaved {sum(len(v) for v in all_candidates.values())} candidates -> {OUT}")


if __name__ == "__main__":
    main()
