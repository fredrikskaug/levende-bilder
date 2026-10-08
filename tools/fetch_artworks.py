"""Hent malerier fra Nasjonalmuseets IIIF-tjeneste.

For hvert verk i artworks.json:
  1. Hent verkssiden (samlingen/objekt/<id>) og les metadata (JSON-LD) og bildelisten
     (originalFile) fra sidens props.
  2. Bygg IIIF-URL på samme måte som ImageUtilHelperService.GetIiifImageUrl i NamWeb.
  3. Last ned bildet i ønsket bredde til web/art/<slug>/image.jpg.

Resultatet skrives til web/art/works.json, som visningen i nettleseren leser.

    python tools/fetch_artworks.py                 # alle verk i artworks.json, 1500 px
    python tools/fetch_artworks.py --width 2000
    python tools/fetch_artworks.py --only NG.M.00939
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / "web" / "art"
ARTWORKS_FILE = Path(__file__).resolve().parent / "artworks.json"

OBJECT_PAGE_URL = "https://www.nasjonalmuseet.no/samlingen/objekt/{0}"
# Same values as IiifService in NamWeb.Presentation/appsettings.json
IIIF_IMAGE_URL = "https://ms01.nasjonalmuseet.no/iip/?iiif=/tif/{0}/full/{1},{2}/0/default.jpg"
MUNCH_ASSET_URL = "https://www.munch.no/globalassets/"

USER_AGENT = "Mozilla/5.0 (DepthGenerator kodekveld; +https://www.nasjonalmuseet.no)"


def get_iiif_image_url(original_file: str, width: int | None = 2000, height: int | None = None) -> str:
    """Port of ImageUtilHelperService.GetIiifImageUrl (WebUtility.UrlEncode ~ quote_plus)."""
    return IIIF_IMAGE_URL.format(
        urllib.parse.quote_plus(original_file),
        "" if width is None else str(width),
        "" if height is None else str(height),
    )


def http_get(url: str, retries: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as res:
                return res.read()
        except Exception as e:  # noqa: BLE001 - retry on any network error
            last_error = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} feilet: {last_error}")


def slugify(object_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", object_id.lower()).strip("-")


def as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def parse_json_ld(html: str) -> dict:
    for m in re.finditer(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', html, re.S):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("@type") == "CreativeWork":
            return data
    return {}


def parse_images(html: str) -> tuple[list[dict], str]:
    """Return the object's image list (with originalFile) and its copyright field."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r'"images":\[', html):
        start = m.end() - 1
        try:
            images, end = decoder.raw_decode(html, start)
        except json.JSONDecodeError:
            continue
        if images and isinstance(images[0], dict) and "originalFile" in images[0]:
            cm = re.match(r'\s*,\s*"copyright"\s*:\s*', html[end:])
            copyright_text = ""
            if cm:
                value, _ = decoder.raw_decode(html, end + cm.end())
                copyright_text = re.sub(r"<[^>]+>", "", value or "").strip()
            return images, copyright_text
    return [], ""


def pick_image(images: list[dict], object_id: str, override: str | None) -> dict | None:
    if override:
        return next((i for i in images if i.get("originalFile") == override), {"originalFile": override})
    usable = [i for i in images if i.get("originalFile") and not i.get("is360") and not i.get("panorama")]
    # Gigapixel scans are often shot with the frame; the regular image is cropped to the canvas.
    regular = [i for i in usable if not i.get("isGigapixel")]
    # Detail shots get numeric names (e.g. 54318.tif); the main reproduction is usually named after the object
    named = [i for i in regular if i["originalFile"].startswith(object_id)]
    return (named or regular or usable or [None])[0]


def save_image(data: bytes, slug: str, crop: list[float] | None) -> None:
    """Write image.jpg, optionally cropping away mounts/borders: crop = [left, top, right, bottom] fractions."""
    out_dir = ART_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "image.jpg"
    path.write_bytes(data)
    if crop:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            box = (round(crop[0] * w), round(crop[1] * h), round(w - crop[2] * w), round(h - crop[3] * h))
            im.convert("RGB").crop(box).save(path, quality=92)


def fetch_work(entry: dict, width: int) -> dict:
    if entry.get("source") == "munchmuseet":
        return fetch_munchmuseet(entry, width)
    return fetch_nasjonalmuseet(entry, width)


def fetch_munchmuseet(entry: dict, width: int) -> dict:
    """Munch's own photographs aren't in Nasjonalmuseet's collection. MUNCH publishes them as article
    images, so the metadata comes from artworks.json and the file from munch.no's image resizer."""
    slug = slugify(entry["id"])
    url = f"{MUNCH_ASSET_URL}{entry['file']}?w={width}&mode=Max&quality=95"
    save_image(http_get(url), slug, entry.get("crop"))
    return {
        "id": entry["id"],
        "slug": slug,
        "title": entry["title"],
        "titleEn": entry.get("titleEn", ""),
        "artist": "Edvard Munch",
        "creators": [{"name": "Edvard Munch", "url": "https://www.munch.no/edvard-munch/hvem-var-edvard-munch/"}],
        "genre": entry.get("genre", "Fotografi"),
        "date": entry.get("date", ""),
        "material": entry.get("material", ""),
        "institution": "Munchmuseet",
        "license": "© Munchmuseet",
        "photoCredit": "",
        "pageUrl": entry["pageUrl"],
        "originalFile": entry["file"],
        "imageUrl": url,
        "image": f"art/{slug}/image.jpg",
        "depth": f"art/{slug}/depth.png",
        "focus": entry.get("focus", 0.5),
        "strength": entry.get("strength", 1.0),
    }


def fetch_nasjonalmuseet(entry: dict, width: int) -> dict:
    object_id = entry["id"]
    slug = slugify(object_id)
    page_url = OBJECT_PAGE_URL.format(object_id.replace("&", "_"))  # NG.K&H.… → NG.K_H.… like NamWeb
    html = http_get(page_url).decode("utf-8", errors="replace")

    ld = parse_json_ld(html)
    images, copyright_text = parse_images(html)
    image = pick_image(images, object_id, entry.get("file"))
    if not image:
        raise RuntimeError(f"{object_id}: fant ingen bilder på {page_url}")
    if copyright_text:
        print(f"  ! {object_id}: vernet av opphavsrett «{copyright_text}» – bare til intern demo", file=sys.stderr)

    original_file = image["originalFile"]
    iiif_url = get_iiif_image_url(original_file, width=width)
    save_image(http_get(iiif_url), slug, entry.get("crop"))

    # JSON-LD uses a single value or a list depending on how many there are
    creators = [
        {"name": c["name"], "url": c.get("sameAs", "")}
        for c in as_list(ld.get("creator"))
        if isinstance(c, dict) and c.get("name")
    ]
    title_en = next((re.sub(r"\s*\(ENG\)\s*$", "", t) for t in as_list(ld.get("alternateName")) if t.endswith("(ENG)")), "")
    return {
        "id": object_id,
        "slug": slug,
        "title": ld.get("name") or object_id,
        "titleEn": title_en,
        "artist": ", ".join(c["name"] for c in creators),
        "creators": creators,
        "genre": ld.get("genre", ""),
        "date": ld.get("dateCreated", ""),
        "material": ld.get("material", ""),
        "institution": "Nasjonalmuseet",
        # Protected works aren't CC BY: show the rights holder from the object page instead
        "license": copyright_text or "CC BY 4.0",
        "protected": bool(copyright_text),
        "photoCredit": ", ".join(image.get("photocredit") or []),
        "pageUrl": page_url,
        "originalFile": original_file,
        "imageUrl": iiif_url,
        "image": f"art/{slug}/image.jpg",
        "depth": f"art/{slug}/depth.png",
        "focus": entry.get("focus", 0.5),
        "strength": entry.get("strength", 1.0),
    }


MANIFEST_FILE = ART_DIR / "works.json"


def read_manifest() -> list[dict]:
    if not MANIFEST_FILE.exists():
        return []
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))["works"]


def write_manifest(works: list[dict]) -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps({"works": works}, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def add_to_artworks(entry: dict) -> None:
    """Append an entry to artworks.json, keeping the hand-written one-line-per-work layout."""
    text = ARTWORKS_FILE.read_text(encoding="utf-8")
    line = "    " + json.dumps(entry, ensure_ascii=False).replace('{"', '{ "').replace('"}', '" }')
    updated, n = re.subn(r"\n  \]\n\}\s*$", f",\n{line}\n  ]\n}}\n", text)
    if n != 1:  # unexpected layout: fall back to plain JSON
        config = json.loads(text)
        config["works"].append(entry)
        updated = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    ARTWORKS_FILE.write_text(updated, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, default=1500, help="bildebredde fra IIIF (default 1500)")
    parser.add_argument("--only", nargs="*", help="bare disse objekt-ID-ene")
    args = parser.parse_args()

    config = json.loads(ARTWORKS_FILE.read_text(encoding="utf-8"))
    entries = [e for e in config["works"] if not args.only or e["id"] in args.only]
    existing = {w["id"]: w for w in read_manifest()}

    for entry in entries:
        print(f"→ {entry['id']}")
        try:
            work = fetch_work(entry, args.width)
        except Exception as e:  # noqa: BLE001 - keep going with the other works
            print(f"  ✗ {e}", file=sys.stderr)
            continue
        existing[work["id"]] = {**existing.get(work["id"], {}), **work}
        print(f"  ✓ {work['title']} – {work['artist']} ({work['date']})  [{work['originalFile']}]")

    # Only what artworks.json lists, in that order (keeps earlier data if a fetch failed this time)
    works = [existing[e["id"]] for e in config["works"] if e["id"] in existing]
    write_manifest(works)
    print(f"\nSkrev {MANIFEST_FILE.relative_to(ROOT)} ({len(works)} verk)")


if __name__ == "__main__":
    main()
