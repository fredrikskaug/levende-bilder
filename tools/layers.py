"""Bakgrunnsplate bak forgrunnskantene, laget med LaMa.

Når perspektivet flytter seg, kan man se litt bak nære ting (et hode foran sjøen). Med bare
ett dybdekart finnes det ingenting der, og skyggeleggeren strekker kantpikslene. Her deles
bildet i to lag, som i 3D Photo Inpainting (Shih m.fl. 2020) og SLIDE (Jampani m.fl. 2021):

- forgrunn: et bånd innenfor kanten av alt som står foran noe annet (der dybden faller brått)
- bakgrunn: hele bildet, men med båndet fylt inn av LaMa, og dybden der fortsatt fra
  bakgrunnssiden, så sjøen fortsetter inn under hodet

Skyggeleggeren går under forgrunnen når synslinjen kommer inn fra siden under kanten, og
treffer platen i stedet for å strekke kanten. Står du rett foran, vises bare originalen.

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

MAX_SHIFT = 0.05   # the largest parallax we fill for, as a share of the image width (the defaults give ~3 %)
EDGE_PX = 3        # a depth edge drops by EDGE_JUMP within this many pixels (the depth map is softened)
EDGE_JUMP = 0.05   # in 0–1 nearness
GROW_PX = 2        # also fill the antialiased edge pixels, which mix foreground and background colours
# LaMa fills from whatever surrounds the hole. Given only the band, it continues the head into
# it as much as the lake. So it gets a wider hole, which swallows thin things like people
# whole, and fills from the background alone; only the band of its result is kept.
REMOVE_SCALE = 3
REMOVE_MIN_PX = 12
# The viewer only reads the plate inside the bands. Elsewhere it's one flat colour, which JPEG
# shrinks to almost nothing; the margin keeps texture filtering at the band edges clean.
PLATE_MARGIN_PX = 8


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


def min_filter(a: np.ndarray, r: int) -> np.ndarray:
    """Square min filter, separably."""
    t = -torch.from_numpy(np.ascontiguousarray(a))[None, None]
    k = 2 * r + 1
    t = F.max_pool2d(F.pad(t, (r, r, 0, 0), mode="replicate"), (1, k), stride=1)
    t = F.max_pool2d(F.pad(t, (0, 0, r, r), mode="replicate"), (k, 1), stride=1)
    return -t[0, 0].numpy()


def split(near: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From the depth map: the front band (bool), the depth behind it (0–1 nearness), and
    the wider area for LaMa to fill (bool).

    An edge can uncover as much as the parallax moves its two sides apart: MAX_SHIFT of the
    width times the depth jump. So the band behind a head against a distant lake is wide, and
    the one behind a fold in the ground is narrow or nothing."""
    from scipy import ndimage

    n = near.astype(np.float32)
    drop = n - min_filter(n, EDGE_PX)            # how far the depth falls within a few pixels
    edges = drop > EDGE_JUMP                     # the near side of a sudden drop
    if not edges.any():
        empty = np.zeros(n.shape, bool)
        return empty, n, empty
    dist, (iy, ix) = ndimage.distance_transform_edt(~edges, return_indices=True)
    jump = drop[iy, ix]                          # of the nearest edge
    behind = n[iy, ix] - jump                    # the depth at the foot of that edge
    reach = MAX_SHIFT * n.shape[1] * jump
    near_side = n - behind > 0.5 * jump
    front = ndimage.binary_dilation((dist <= reach) & near_side, iterations=GROW_PX)
    remove = (dist <= REMOVE_SCALE * reach + REMOVE_MIN_PX) & near_side
    remove = ndimage.binary_dilation(remove, iterations=GROW_PX) | front
    # Islands left inside (a chest whose depth differs a little from the outline) would be the
    # only context LaMa sees there, and it paints the person right back in
    remove = ndimage.binary_fill_holes(remove)
    # What's behind the band continues from the foot of the edge, not from the foreground itself
    behind = ndimage.gaussian_filter(behind, 2)
    back = np.where(front, np.minimum(behind, n), n)
    return front, np.clip(back, 0, 1), remove


@torch.no_grad()
def inpaint(lama, rgb: np.ndarray, mask: np.ndarray, device: torch.device) -> np.ndarray:
    """LaMa on the whole image (it handles ~2k px); returns uint8 RGB with only the mask changed."""
    h, w = mask.shape
    image = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(device)
    m = torch.from_numpy(mask.astype(np.float32))[None, None].to(device)
    ph, pw = (-h) % 8, (-w) % 8  # the network wants multiples of 8
    image_p = F.pad(image, (0, pw, 0, ph), mode="reflect")
    m_p = F.pad(m, (0, pw, 0, ph), mode="constant", value=0)
    out = lama(image_p, m_p)[..., :h, :w].clamp(0, 1)
    out = out * m + image * (1 - m)
    return (out[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def make_layers(lama, image: Image.Image, depth: Image.Image, device: torch.device) -> tuple[Image.Image, Image.Image, float]:
    """Background plate (RGB) and layers map (R: depth behind, G: 255 in the front band).

    Also returns how much of the image was filled in (0–1)."""
    rgb = np.asarray(image.convert("RGB"))
    near = np.asarray(depth.convert("L"), dtype=np.float32) / 255
    front, back, remove = split(near)
    from scipy import ndimage

    plate = np.empty_like(rgb)
    plate[:] = rgb.reshape(-1, 3).mean(0).astype(np.uint8)
    if front.any():
        filled = inpaint(lama, rgb, remove, device)
        margin = ndimage.binary_dilation(front, iterations=PLATE_MARGIN_PX)
        plate[margin] = rgb[margin]
        plate[front] = filled[front]
    layers = np.zeros((*front.shape, 3), dtype=np.uint8)
    layers[..., 0] = np.round(back * 255).astype(np.uint8)
    layers[..., 1] = front.astype(np.uint8) * 255
    return Image.fromarray(plate), Image.fromarray(layers), float(front.mean())
