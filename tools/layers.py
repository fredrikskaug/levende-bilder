"""Dybdelag med KI-utfylt bakgrunn, slik at perspektivet kan flyttes uten strukne kanter.

Med ett dybdekart strekkes kanten mellom en båt og sjøen bak den ut når perspektivet flytter
seg: det finnes ingenting bak båten å vise. Her deles maleriet i lag der dybden hopper (himmel og
fjerne fjell, åser og strand, båt og folk …), omtrent som i 3D Photo Inpainting (Shih m.fl. 2020)
og SLIDE (Jampani m.fl. 2021). Hvert lag har sine egne piksler (maleriets), det som er skjult bak
nærmere lag (fylt inn av LaMa: sjøen bak båten, fjellet bak hodene) og ingenting ellers, og én
jevn dybdeflate. Kanten er lagets omriss i alfakartet, så den flyttes hel med laget. Skyggeleggeren
følger synslinjen gjennom lagene forfra og bakover og viser det første laget som har noe der; i
ro er det alltid maleriets egen piksel.

- Lagene finnes for hvert bilde: grensene legges midt i de dybdehoppene som skiller mest, så
  et landskap med mange plan får flere lag enn et portrett.
- Grensene følger fargekantene i maleriet (guided filter), ikke det uskarpe dybdekartet.
- Det skjulte fylles bare inn så langt fra lagets egne piksler som den største parallaksen mellom
  dem rekker.

LaMa (Suvorov m.fl. 2022, Apache 2.0) er en rask utfyllingsmodell som er god på flater som
vann, himmel og bakker. Vi bruker TorchScript-versjonen fra IOPaint.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
LAMA_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"

MAX_LAYERS = 7
MAX_SHIFT = 0.12      # the largest parallax to fill for, as a share of the width (strength 8 % is 9.6 %)
EDGE_PX = 3           # a depth edge drops by EDGE_JUMP within this many pixels
EDGE_JUMP = 0.04      # in 0–1 nearness
MIN_SEPARATION = 0.06 # between two layer boundaries, in nearness
MIN_SHARE = 0.03      # a boundary must separate this share of all depth-edge strength
MIN_AREA = 0.002      # islands smaller than this share of the picture join their surroundings
PIECE_AREA = 0.01     # pieces smaller than this that continue into another layer without a step join it
SLACK = 0.04          # how far outside its layer's range a pixel's depth may be before it's taken as misplaced (a thin raised bow)
TEAR = 0.1            # smaller steps than this stay joined, and stretch a little instead (a head and its body)
SNAP_PX = 6           # soft steps in the depth map are made sharp within this radius …
GUIDE_PX = 4          # … and then moved onto the nearest edge in the colours, at most this far
GROW_PX = 3           # nearer layers take over this many pixels across a depth edge (their rim), which they let go as they move
RIM_PX = 5            # pixels this close to a depth edge get their depth from their layer's interior …
CONTEXT_PX = 2        # … and LaMa doesn't see the ones this close to something in front, so it doesn't continue it
MARGIN_PX = 12        # what's behind is filled in this far beyond where the largest parallax can reveal it
PAD_PX = 60           # LaMa fills this much more than can be seen: what's in front, further in, it may continue
NARROW_PX = 10        # parts of the hole narrower than twice this (a spire, a rifle) …
ROUND_PX = 8          # … reach this far out into the background, so LaMa doesn't see their shape as an object
# Alpha maps: own pixels, own rim (taken over from what's behind), AI-filled, nothing (0). The rim
# isn't left to what's behind: it's a mix of both, and left behind it would trace the figure's outline.
ALPHA_OWN, ALPHA_RIM, ALPHA_FILLED = 255, 220, 160


def lama_path() -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints" / "big-lama.pt"


def load_lama(device: torch.device):
    path = lama_path()
    if not path.exists():
        print(f"Laster ned LaMa ({LAMA_URL}, ~200 MB) …")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(LAMA_URL, str(path))
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        if md5 != LAMA_MD5:
            path.unlink()
            raise RuntimeError(f"LaMa-filen har feil sjekksum ({md5}), slettet den")
    model = torch.jit.load(str(path), map_location=device)
    return model.eval()


@torch.no_grad()
def inpaint(lama, rgb: np.ndarray, hole: np.ndarray, device: torch.device) -> np.ndarray:
    """LaMa on the whole image (it handles ~2k px); uint8 RGB with only the hole changed."""
    h, w = hole.shape
    image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().div(255).to(device)
    m = torch.from_numpy(hole.astype(np.float32))[None, None].to(device)
    ph, pw = (-h) % 8, (-w) % 8  # the network wants multiples of 8
    out = lama(F.pad(image, (0, pw, 0, ph), mode="reflect"), F.pad(m, (0, pw, 0, ph)))[..., :h, :w].clamp(0, 1)
    out = out * m + image * (1 - m)
    return (out[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def min_filter(a: np.ndarray, r: int) -> np.ndarray:
    t = -torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None]
    k = 2 * r + 1
    t = F.max_pool2d(F.pad(t, (r, r, 0, 0), mode="replicate"), (1, k), stride=1)
    t = F.max_pool2d(F.pad(t, (0, 0, r, r), mode="replicate"), (k, 1), stride=1)
    return -t[0, 0].numpy()


def snap_edges(n: np.ndarray) -> np.ndarray:
    """The depth with its soft steps made sharp: each pixel in a step goes to the side it's closest to.

    The depth model sees the picture at a lower resolution, so the step from a figure to the
    mountain behind is spread over a few pixels. The depths in between belong to neither, and
    would otherwise become thin rings of layers of their own around the figure."""
    lo, hi = min_filter(n, SNAP_PX), -min_filter(-n, SNAP_PX)
    step = hi - lo >= TEAR  # a gentle slope (a mountainside receding towards the ridge) stays as it is
    return np.where(step, np.where(n - lo > hi - n, hi, lo), n)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Smoothing of src that follows the edges in a colour image (He m.fl. 2013)."""
    import cv2

    mean = lambda x: cv2.blur(x, (2 * r + 1, 2 * r + 1))  # noqa: E731
    mi, ms = mean(guide), mean(src)
    cov = mean(guide * src[..., None]) - mi * ms[..., None]
    var = np.empty(guide.shape[:2] + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            var[..., i, j] = var[..., j, i] = mean(guide[..., i] * guide[..., j]) - mi[..., i] * mi[..., j]
    var += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(var, cov[..., None])[..., 0]
    b = ms - (a * mi).sum(-1)
    return (mean(a) * guide).sum(-1) + mean(b)


def disk(r: int) -> np.ndarray:
    return np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r


def smooth_fill(values: np.ndarray, known: np.ndarray, iterations: int = 24) -> np.ndarray:
    """Fill in the unknown pixels smoothly from the known ones, which stay as they are.

    Like a membrane stretched over the known values: push-pull on an image pyramid gives a first
    guess, and on the way back up each level is relaxed towards the average of its neighbours."""
    if not known.any():
        return np.full(values.shape, float(values.mean()), np.float32)
    w = torch.from_numpy(known.astype(np.float32))[None, None]
    vw = torch.from_numpy(np.where(known, values, 0).astype(np.float32))[None, None]
    levels = [(vw, w)]
    while min(levels[-1][0].shape[-2:]) > 8:
        vw, w = levels[-1]
        levels.append((F.avg_pool2d(vw, 2, ceil_mode=True), F.avg_pool2d(w, 2, ceil_mode=True)))
    vw, w = levels[-1]
    est = torch.where(w > 0, vw / w.clamp_min(1e-8), vw.sum() / w.sum())
    for vw, w in reversed(levels[:-1]):
        est = F.interpolate(est, size=vw.shape[-2:], mode="bilinear", align_corners=False)
        fixed, data = w > 0, vw / w.clamp_min(1e-8)
        est = torch.where(fixed, data, est)
        for _ in range(iterations):
            est = torch.where(fixed, data, F.avg_pool2d(F.pad(est, (1, 1, 1, 1), mode="replicate"), 3, stride=1))
    return est[0, 0].numpy()


def boundaries(n: np.ndarray) -> list[float]:
    """Depths to split the picture at: in the middle of the depth edges that separate the most.

    Each edge pixel (where the depth drops suddenly) has a near and a far side; a boundary
    between them separates that edge. Greedily take the boundary that separates the most edge
    strength not yet separated, until it's little or there are MAX_LAYERS layers."""
    drop = n - min_filter(n, EDGE_PX)
    edges = drop > EDGE_JUMP
    near, jump = n[edges], drop[edges]
    far = near - jump
    total = jump.sum()
    if total <= 0:
        return []
    candidates = np.arange(0.03, 0.97, 0.01)
    # Only the middle half of an edge counts, so the boundary lands mid-way between its sides
    inside = (far[None, :] + 0.25 * jump[None, :] < candidates[:, None]) & (candidates[:, None] < near[None, :] - 0.25 * jump[None, :])
    weight = jump[None, :] * inside
    chosen: list[float] = []
    open_edges = np.ones(len(jump), bool)
    while len(chosen) < MAX_LAYERS - 1:
        score = (weight * open_edges[None, :]).sum(axis=1)
        for t in chosen:
            score[np.abs(candidates - t) < MIN_SEPARATION] = 0
        best = int(score.argmax())
        if score[best] < MIN_SHARE * total:
            break
        chosen.append(float(candidates[best]))
        open_edges &= ~inside[best]
    return sorted(chosen)


def assign(rgb: np.ndarray, n: np.ndarray, cuts: list[float]) -> np.ndarray:
    """Layer index per pixel (0 = farthest).

    Each boundary is moved onto the edge in the painting nearby: the depth model's edges lag a
    few pixels behind the real ones, and on a mountain ridge the depth slopes off gradually, but
    the colours know where the mountain ends and the sky begins. Then small islands join the
    neighbouring layer closest in depth: a head that's a little farther away than its body joins
    the body, not the mountain around it."""
    from scipy import ndimage

    guide = rgb.astype(np.float32) / 255
    labels = np.zeros(n.shape, np.int32)
    nearer = np.ones(n.shape, bool)
    for cut in cuts:
        mask = n >= cut
        # Only a few pixels either way: where a figure's colours are close to what's behind it
        # (dark hair against dark trees) the colours alone would cut right into it
        rim = ndimage.binary_dilation(mask, iterations=GUIDE_PX) & ~ndimage.binary_erosion(mask, iterations=GUIDE_PX)
        moved = guided_filter(guide, mask.astype(np.float32), GUIDE_PX, 1e-3) > 0.5
        nearer &= np.where(rim, moved, mask)
        labels += nearer
    min_area = MIN_AREA * labels.size
    h, w = labels.shape
    for k in range(len(cuts) + 1):
        parts, count = ndimage.label(labels == k)
        sizes = np.bincount(parts.ravel(), minlength=count + 1)
        for i, box in enumerate(ndimage.find_objects(parts), start=1):
            if box is None or sizes[i] >= min_area:
                continue
            box = tuple(slice(max(s.start - 3, 0), min(s.stop + 3, size)) for s, size in zip(box, (h, w)))
            part = parts[box] == i
            ring = ndimage.binary_dilation(part, iterations=2) & ~part
            if not ring.any():
                continue
            depth, near, around = np.median(n[box][part]), n[box][ring], labels[box][ring]
            options = np.unique(around)
            labels[box][part] = options[np.argmin([abs(np.median(near[around == m]) - depth) for m in options])]
    join_pieces(n, labels, len(cuts) + 1)
    return labels


def join_pieces(n: np.ndarray, labels: np.ndarray, count: int) -> None:
    """Pieces of one thing that a boundary cuts through join the part they continue into.

    The top of a head can be a little nearer than the face, and a boundary between them would
    make the layer with the face have a hole where the top of the head was, sloping off to the
    background, which stretches as the head moves. A piece that meets one neighbouring layer
    without a depth step along a good part of its rim joins it, unless it's big. A strip of a
    lake between two boundaries meets the layers on both sides without a step: it's a slice of
    one surface, and stays."""
    from scipy import ndimage

    h, w = labels.shape
    for k in range(count):
        parts, found = ndimage.label(labels == k)
        sizes = np.bincount(parts.ravel(), minlength=found + 1)
        for i, box in enumerate(ndimage.find_objects(parts), start=1):
            if box is None or sizes[i] >= PIECE_AREA * labels.size:
                continue
            box = tuple(slice(max(s.start - 4, 0), min(s.stop + 4, size)) for s, size in zip(box, (h, w)))
            part = parts[box] == i
            ring = ndimage.binary_dilation(part, iterations=2) & ~part
            around, nb = labels[box], n[box]
            smooth = []  # (how much of the rim, layer) for the layers it continues into
            for m in np.unique(around[ring]):
                touching = ring & (around == m)
                inside = ndimage.binary_dilation(touching, iterations=2) & part
                if touching.sum() > 0.1 * ring.sum() and inside.any() and abs(np.median(nb[touching]) - np.median(nb[inside])) < TEAR:
                    smooth.append((touching.sum(), m))
            if len(smooth) == 1 and smooth[0][0] > 0.2 * ring.sum():
                around[part] = smooth[0][1]


def crisp_depth(near: np.ndarray, sharp: np.ndarray, labels: np.ndarray, cuts: list[float]) -> np.ndarray:
    """The depth with a clean step wherever two layers meet at a depth edge.

    The depth map is soft at edges: the rim of a figure slopes down towards the background, and
    the background rises towards the figure. Taken as it is, both would be stretched when the
    perspective moves. Instead each layer's depth near an edge is continued from its interior, so
    the step is sharp and lands on the layer boundary.
    Thin things the layers missed (a raised fiddle bow) are flattened into their layer the same
    way. Where a surface crosses a boundary without an edge (a lake), the depth stays as it is."""
    import cv2
    from scipy import ndimage

    mean = lambda x: cv2.blur(x, (2 * RIM_PX + 1, 2 * RIM_PX + 1))  # noqa: E731
    bounds = [0.0, *cuts, 1.0]
    out = near.copy()
    for m in range(len(cuts) + 1):
        mine = labels == m
        if not mine.any():
            continue
        # The layer's range: its boundaries, or wider where a good part of it is (a piece that
        # joined it); only stray pixels outside are taken as misplaced
        spread = np.percentile(sharp[mine], [2, 98])
        lo, hi = min(bounds[m], spread[0]), max(bounds[m + 1], spread[1])
        # Other layers' pixels that are a step away from this layer's depth right next to them
        share = mean(mine.astype(np.float32))
        local = mean(np.where(mine, sharp, 0).astype(np.float32)) / np.maximum(share, 1e-6)
        edge = ~mine & (share > 0) & (np.abs(sharp - local) >= TEAR)
        core = mine & ~ndimage.binary_dilation(edge, iterations=RIM_PX) & (near > lo - SLACK) & (near < hi + SLACK)
        if core.sum() >= 0.1 * mine.sum() > 0:
            out[mine] = smooth_fill(near, core)[mine]
    return out


def make_layers(lama, image: Image.Image, near: np.ndarray, device: torch.device) -> tuple[list[dict], dict]:
    """Split into depth layers and fill in behind them.

    near: 0–1 nearness at image size, not dilated. Returns the layers back to front, each
    {color, alpha, depth} as PIL images, and some numbers for the log."""
    from scipy import ndimage

    rgb = np.array(image.convert("RGB"))
    near = near.astype(np.float32)
    sharp = snap_edges(near)
    cuts = boundaries(sharp)
    exact = assign(rgb, sharp, cuts)
    labels = ndimage.grey_dilation(exact, footprint=disk(GROW_PX))
    rim = labels != exact  # taken over from what's behind: the soft edge of the depth map
    count = len(cuts) + 1
    depth = crisp_depth(near, sharp, labels, cuts)
    shift = MAX_SHIFT * max(near.shape)  # pixels a point moves per unit of depth difference, at most

    layers = []
    for k in range(count):
        own = labels == k
        nearer = labels > k
        # One smooth surface: the layer's own depth, continued under everything else. Its edge is
        # its outline in the alpha map, so it moves as a whole with the layer and keeps its shape.
        d = smooth_fill(depth, own)
        # What's in front of the layer and can move aside: as far in from the layer's own pixels as
        # the parallax between the two can reach, and a little more
        reach = shift * np.maximum(depth - d, 0) + MARGIN_PX
        filled = nearer & (ndimage.distance_transform_edt(~own) <= reach)
        if not filled.any():
            color = np.full(rgb.shape, 128, np.uint8)  # its pixels are all its own, drawn from the image
        else:
            # LaMa fills in what was behind what's in front, but only that band and a margin around
            # it; the rest of the painting, also what's in front further away, is its context. One
            # big hole (behind the farthest layer, half the picture) leaves it guessing, and it
            # invents things, like a bush behind a church spire, where with the forest around the
            # church in view it continues the forest. It doesn't see the pixels right next to
            # what's removed, where some of its colour may be left.
            # Where the hole is narrow and pointed, it reaches a little out into the background too:
            # given a hole shaped like a church spire, LaMa paints a tree in it. Not around broad
            # shapes like a head, where it needs the background right next to it to match. Those
            # pixels of the background are only support; the painting's own are shown there.
            context = (labels <= k) & ~ndimage.binary_dilation(nearer, iterations=CONTEXT_PX)
            hole = ~context & ndimage.binary_dilation(filled, iterations=PAD_PX)
            narrow = hole & ~ndimage.binary_opening(hole, structure=disk(NARROW_PX))
            hole |= ndimage.binary_dilation(narrow, structure=disk(ROUND_PX))
            color = inpaint(lama, rgb, hole, device)
            # Flat grey where it's never shown, so the JPEG is small
            color = np.where(ndimage.binary_dilation(filled, iterations=4)[..., None], color, 128).astype(np.uint8)
        # Its own pixels are the painting's, drawn from the image; only what was hidden is filled in
        alpha = np.where(own, np.where(rim, ALPHA_RIM, ALPHA_OWN), np.where(filled, ALPHA_FILLED, 0))
        alpha = alpha.astype(np.uint8)
        layers.append({
            "color": Image.fromarray(color),
            "alpha": Image.fromarray(alpha, mode="L"),
            "depth": Image.fromarray(np.round(np.clip(d, 0, 1) * 255).astype(np.uint8), mode="L"),
            "share": float(own.mean()),
            "filled": float(filled.mean()),
        })
    info = {"cuts": cuts, "layers": [(round(l["share"] * 100, 1), round(l["filled"] * 100, 1)) for l in layers]}
    return layers, info
