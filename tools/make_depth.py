"""Lag dybdekart med Depth Anything 3 for alle verk i web/art/works.json.

Modellen er DA3MONO-LARGE (Apache 2.0), den monokulære varianten av Depth Anything 3
som er laget for relativ dybde fra ett enkelt bilde. Hovedserien (DA3-LARGE o.l.) er
laget for geometri fra flere bilder, og ser et maleri som en flat flate.

Resultatet lagres som en 8-bits gråtone-PNG ved siden av bildet. Hvit betyr nær
og svart betyr langt unna. I tillegg lages en bakgrunnsplate (background.jpg) og et lagkart
(layers.png) med LaMa, slik at man ser bakgrunn og ikke strukne piksler bak nære ting når
perspektivet flytter seg (se layers.py).

    python tools/make_depth.py                         # alle verk
    python tools/make_depth.py --only NG.M.00939       # bare ett
    python tools/make_depth.py --only-layers           # bare bakgrunnsplatene, fra dybdekartene som finnes
    python tools/make_depth.py --mapping disparity --res 756
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))  # layers.py, also when run with python -I
ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
MANIFEST = WEB_DIR / "art" / "works.json"

# Raw sky probability from the last forward pass (see keep_raw_sky)
_last_sky: dict[str, np.ndarray | None] = {"sky": None}


def keep_raw_sky() -> None:
    """Skip DA3MONO's hard sky threshold and keep the probability instead.

    The model pushes pixels with sky probability >= 0.3 to max depth. Painted skies often
    land just around that value, which leaves dark holes in the sky; elsewhere the raw
    depth places the sky *in front of* the mountains. soften_sky() blends gradually instead.
    """
    from depth_anything_3.model.da3 import DepthAnything3Net

    def capture(self, output):
        _last_sky["sky"] = output.sky.float().squeeze().cpu().numpy() if "sky" in output else None
        return output

    DepthAnything3Net._process_mono_sky_estimation = capture


def load_model(model_id: str, device: torch.device):
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_id)
    return model.to(device=device).eval()


def predict_depth(model, image: np.ndarray, res: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Relative depth (larger = farther) and sky probability, at processing resolution."""
    _last_sky["sky"] = None
    pred = model.inference([image], process_res=res, process_res_method="upper_bound_resize")
    return pred.depth[0].astype(np.float32), _last_sky["sky"]


def gaussian_kernel(sigma: float) -> torch.Tensor:
    radius = max(1, int(round(sigma * 3)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur of a (1, 1, H, W) tensor."""
    k = gaussian_kernel(sigma)
    pad = len(k) // 2
    t = F.conv2d(F.pad(t, (pad, pad, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(t, (0, 0, pad, pad), mode="replicate"), k.view(1, 1, -1, 1))


def soften_sky(depth: np.ndarray, sky: np.ndarray | None, lo: float, hi: float, horizon: float = 0.6) -> np.ndarray:
    """Push likely sky towards the far plane, weighted by a smoothed sky probability."""
    if sky is None or sky.shape != depth.shape:
        return depth
    s = blur(torch.from_numpy(sky)[None, None], sigma=depth.shape[1] * 0.012)[0, 0].numpy()
    w = np.clip((s - lo) / (hi - lo), 0.0, 1.0)
    w = w * w * (3 - 2 * w)
    ground = depth[s < lo]
    if ground.size < 100:
        return depth
    # In a landscape, whatever is above sky is also sky. DA3 often misses the top of painted
    # skies, so spread the weight upwards. Only from the upper part, so sky reflected in
    # water doesn't drag the mountains above it backwards.
    source = w.copy()
    source[int(source.shape[0] * horizon):] = 0
    upward = torch.from_numpy(np.ascontiguousarray(np.maximum.accumulate(source[::-1], axis=0)[::-1]))[None, None]
    # Patchy detections become vertical stripes; a horizontal closing fills the gaps between them
    k = max(3, int(depth.shape[1] * 0.1) | 1)
    upward = F.max_pool2d(upward, (1, k), stride=1, padding=(0, k // 2))
    upward = -F.max_pool2d(-upward, (1, k), stride=1, padding=(0, k // 2))
    upward = blur(upward.float(), sigma=depth.shape[1] * 0.015)[0, 0].numpy()
    w = np.maximum(w, upward)
    far = np.percentile(ground, 99)
    return depth * (1 - w) + np.maximum(depth, far) * w


def to_nearness(depth: np.ndarray, mapping: str) -> np.ndarray:
    """Map depth to a value that grows towards the viewer, before normalisation."""
    depth = np.maximum(depth, 1e-6)
    if mapping == "disparity":  # physically correct parallax, but squashes landscapes flat
        return 1.0 / depth
    if mapping == "log":  # good middle ground for landscapes with a deep background
        return -np.log(depth)
    return -depth  # linear


def normalise(x: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def postprocess(near: np.ndarray, size: tuple[int, int], dilate: int, sigma: float) -> np.ndarray:
    """Upsample to image size, grow the foreground slightly and soften edges.

    Growing near values a few pixels (max filter) makes the depth edge sit just *outside* the
    foreground object, so the object moves as a whole and the stretching lands on the background,
    where it's less noticeable.
    """
    w, h = size
    t = torch.from_numpy(near.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
    if dilate > 0:
        t = F.max_pool2d(t, kernel_size=2 * dilate + 1, stride=1, padding=dilate)
    if sigma > 0:
        t = blur(t, sigma)
    return t[0, 0].numpy()


def estimate(model, rgb: np.ndarray, args) -> np.ndarray:
    """Normalised nearness, 0 far – 1 near."""
    depth, sky = predict_depth(model, rgb, args.res)
    if args.sky == "soft":
        depth = soften_sky(depth, sky, args.sky_lo, args.sky_hi)
    return normalise(to_nearness(depth, args.mapping))


def depth_map(model, image: Image.Image, args) -> Image.Image:
    """The full pipeline for one image: estimate (+ mirrored pass), post-process, 8-bit PNG-ready."""
    rgb = np.asarray(image.convert("RGB"))
    near = estimate(model, rgb, args)
    if args.tta:
        # Relative depth is only defined up to scale/shift; both are normalised before averaging
        flipped = estimate(model, np.ascontiguousarray(rgb[:, ::-1]), args)
        near = normalise(0.5 * (near + flipped[:, ::-1]), 0, 100)
    out = postprocess(near, image.size, args.dilate, args.blur)
    return Image.fromarray(np.round(out * 255).astype(np.uint8))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="depth-anything/DA3MONO-LARGE", help="Hugging Face-modell")
    parser.add_argument("--res", type=int, default=1008, help="prosesseringsoppløsning, lengste side (multiplum av 14)")
    parser.add_argument("--mapping", choices=["log", "disparity", "linear"], default="log")
    parser.add_argument("--sky", choices=["soft", "off"], default="soft", help="myk himmelhåndtering eller rå dybde")
    parser.add_argument("--sky-lo", type=float, default=0.08, help="sky-sannsynlighet der himmelen begynner å skyves bakover")
    parser.add_argument("--sky-hi", type=float, default=0.35, help="sky-sannsynlighet som regnes som ren himmel")
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True,
                        help="snitt av original og speilvendt bilde (jevnere kart)")
    parser.add_argument("--dilate", type=int, default=3, help="piksler forgrunnen vokses med")
    parser.add_argument("--blur", type=float, default=1.2, help="gaussisk uskarphet (sigma, piksler)")
    parser.add_argument("--layers", action=argparse.BooleanOptionalAction, default=True,
                        help="bakgrunnsplate med LaMa bak forgrunnskantene")
    parser.add_argument("--only-layers", action="store_true", help="ikke lag dybdekartene på nytt, bare platene")
    parser.add_argument("--only", nargs="*", help="bare disse objekt-ID-ene")
    return parser


def save_layers(lama, work: dict, image: Image.Image, depth: Image.Image, device: torch.device) -> float:
    """Background plate and layers map next to the image; returns the share that was filled in."""
    import layers

    plate, layer_map, share = layers.make_layers(lama, image, depth, device)
    folder = (WEB_DIR / work["image"]).parent
    plate.save(folder / "background.jpg", quality=90, optimize=True)
    layer_map.save(folder / "layers.png", optimize=True)
    base = work["image"].rsplit("/", 1)[0]
    work["background"] = f"{base}/background.jpg"
    work["layers"] = f"{base}/layers.png"
    return share


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # ✓ and × also when the console isn't UTF-8
    args = build_parser().parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    works = [w for w in manifest["works"] if not args.only or w["id"] in args.only]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if not args.only_layers:
        print(f"Laster {args.model} på {device} …")
        keep_raw_sky()
        model = load_model(args.model, device)
    lama = None
    if args.layers or args.only_layers:
        import layers

        lama = layers.load_lama(device)

    for work in works:
        t0 = time.time()
        image = Image.open(WEB_DIR / work["image"]).convert("RGB")
        if model:
            depth_map(model, image, args).save(WEB_DIR / work["depth"], optimize=True)
            work["depthModel"] = args.model.split("/")[-1]
            work.pop("depthRatio", None)  # from an earlier version
        note = ""
        if lama:
            share = save_layers(lama, work, image, Image.open(WEB_DIR / work["depth"]), device)
            note = f"  plate {share * 100:4.1f} %"
        print(f"  ✓ {work['title']:<32} {image.size[0]}×{image.size[1]}{note}  {time.time() - t0:.1f}s")

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
