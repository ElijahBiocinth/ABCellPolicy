import os
import re
import json
import math
import glob
import sqlite3
import hashlib
import cv2
import numpy as np
import pandas as pd
import imageio
from tqdm import tqdm
from collections import Counter
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image, ImageDraw, ImageFont
import argparse

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
]
DEFAULT_FONT_SIZE = 14
MIN_FONT_SIZE = 8

def find_font_path() -> Optional[str]:
    for path in FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None

FONT_PATH = find_font_path()
if FONT_PATH is None:
    print("[WARN] TTF font not found; fallback to PIL default.")

def parse_args():
    p = argparse.ArgumentParser(description="Render overlays and video from tracking DB.")
    p.add_argument("--scene-name", type=str, default="scene0", help="Label for scene")
    p.add_argument("--db-path", type=str, required=True, help="Path to SQLite DB")
    p.add_argument("--image-dir", type=str, required=True, help="Dir with source images")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory")
    p.add_argument("--fps", type=int, default=30, help="Video FPS")
    p.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha")
    p.add_argument("--contour-thick", type=int, default=3, help="Overlay contour thickness")
    p.add_argument("--px-to-um", type=float, default=0.519933, help="Pixels → µm factor")
    p.add_argument("--make-gif", action="store_true", help="(Ignored) flag for GIF")
    return p.parse_args()

def auto_detect_pattern(image_dir: str, sample_size: int = 200):
    exts = ("png", "tif", "tiff", "jpg", "jpeg")
    files = sum((glob.glob(os.path.join(image_dir, f"*.{e}")) for e in exts), [])
    if not files:
        raise FileNotFoundError(f"No images in {image_dir}")
    sample = sorted(files)[:sample_size]
    for pat in (r'^(.*?T?)(\d+)(\D+\.[A-Za-z0-9]+)$', r'^(.*?)(\d+)(\.[A-Za-z0-9]+)$'):
        parsed = []
        regex = re.compile(pat)
        for p in sample:
            name = os.path.basename(p)
            m = regex.match(name)
            if m:
                pref, nums, suff = m.groups()
                parsed.append((pref, len(nums), suff, int(nums)))
        if parsed:
            break
    if not parsed:
        raise RuntimeError("Cannot detect filename pattern")
    pref = Counter(p[0] for p in parsed).most_common(1)[0][0]
    digs = Counter(p[1] for p in parsed).most_common(1)[0][0]
    suff = Counter(p[2] for p in parsed).most_common(1)[0][0]
    min_idx = min(p[3] for p in parsed)
    return pref, digs, suff, min_idx

def build_filename(prefix, digs, suff, offset, frame):
    idx = frame + offset
    return f"{prefix}{idx:0{digs}d}{suff}"

def stable_color_for_id(uid) -> Tuple[int, int, int]:
    h = hashlib.sha1(str(uid).encode()).hexdigest()[:6]
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    boost = 0.35
    return (
        int((1 - boost) * b + boost * 255),
        int((1 - boost) * g + boost * 255),
        int((1 - boost) * r + boost * 255),
    )

def decode_polygon(js: str) -> Optional[np.ndarray]:
    try:
        pts = json.loads(js)
        arr = np.array(pts, dtype=np.int32)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
    except:
        pass
    return None

def fit_ellipse_metrics(arr_pts: np.ndarray) -> Tuple[float, float]:
    if arr_pts.shape[0] < 5:
        return np.nan, np.nan
    (cx, cy), (w, h), angle = cv2.fitEllipse(arr_pts.astype(np.float32))
    major, minor = (h, w) if h > w else (w, h)
    ecc = math.sqrt(max(0, 1 - (minor / major) ** 2)) if major > 0 else np.nan
    ori = (angle + (90 if h > w else 0)) % 180
    return ecc, ori

def polygon_metrics(arr_pts: np.ndarray, px_to_um2: float, px_to_um: float) -> Tuple[float, float, float, float]:
    poly = Polygon(arr_pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return np.nan, np.nan, np.nan, np.nan
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)
    area_px = poly.area
    peri_px = poly.length
    area = area_px * px_to_um2
    peri = peri_px * px_to_um
    circ = (4 * math.pi * area) / (peri ** 2) if peri > 0 else np.nan
    ecc, ori = fit_ellipse_metrics(np.array(poly.exterior.coords, dtype=np.int32))
    return area, ecc, ori, circ

def mean_orientation_deg(angles: List[float]) -> float:
    ang = np.array([a for a in angles if not np.isnan(a)])
    if ang.size == 0:
        return np.nan
    theta = np.deg2rad(ang) * 2
    mv = np.exp(1j * theta).mean()
    return (np.rad2deg(np.angle(mv)) / 2) % 180

def draw_text_with_bg(
    bgr_img: np.ndarray,
    text: str,
    font_path: Optional[str],
    font_size: int = DEFAULT_FONT_SIZE,
    pos=(8, 8),
    fg=(255, 255, 255, 255),
    bg=(0, 0, 0, 140),
    padding=(8, 5),
    max_width: Optional[int] = None,
    max_height: Optional[int] = None
) -> np.ndarray:
    x, y = pos
    pad_x, pad_y = padding
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)
    current_size = font_size
    if font_path:
        try:
            font = ImageFont.truetype(font_path, current_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    def measure(fnt):
        try:
            bbox = draw.textbbox((0, 0), text, font=fnt)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except AttributeError:
            w, h = draw.textsize(text, font=fnt)
        return w, h
    text_w, text_h = measure(font)
    target_w = None if max_width is None else max_width - 2 * pad_x
    target_h = None if max_height is None else max_height - 2 * pad_y
    if target_w is not None and font_path:
        while text_w < target_w * 0.9 and (target_h is None or text_h < target_h):
            next_size = current_size + 1
            try:
                next_font = ImageFont.truetype(font_path, next_size)
            except Exception:
                break
            next_w, next_h = measure(next_font)
            if target_w is not None and next_w > target_w:
                break
            if target_h is not None and next_h > target_h:
                break
            current_size = next_size
            font = next_font
            text_w, text_h = next_w, next_h
    if target_w is not None:
        while text_w > target_w and current_size > MIN_FONT_SIZE:
            current_size -= 1
            if font_path:
                try:
                    font = ImageFont.truetype(font_path, current_size)
                except Exception:
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()
            text_w, text_h = measure(font)
    if target_h is not None:
        while text_h > target_h and current_size > MIN_FONT_SIZE:
            current_size -= 1
            if font_path:
                try:
                    font = ImageFont.truetype(font_path, current_size)
                except Exception:
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()
            text_w, text_h = measure(font)
    rect_w = int(text_w + 2 * pad_x)
    rect_h = int(text_h + 2 * pad_y)
    corner_radius = max(4, rect_h // 4)
    bg_layer = Image.new("RGBA", (rect_w, rect_h), (0, 0, 0, 0))
    draw_bg = ImageDraw.Draw(bg_layer)
    draw_bg.rounded_rectangle([0, 0, rect_w, rect_h], radius=corner_radius, fill=bg)
    pil_img.alpha_composite(bg_layer, dest=(x, y))
    draw.text((x + pad_x, y + pad_y), text, font=font, fill=fg)
    out_rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    overlay_dir = os.path.join(args.out_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    mp4_path = os.path.join(args.out_dir, f"{args.scene_name}_overlay.mp4")
    conn = sqlite3.connect(args.db_path)
    df = pd.read_sql_query("SELECT time_frame, polygon_points, track_id FROM tracks", conn)
    conn.close()
    frames = sorted(df["time_frame"].unique())
    grouped = df.groupby("time_frame")[["polygon_points", "track_id"]].apply(lambda g: list(zip(g["polygon_points"], g["track_id"])))
    prefix, digs, suff, min_idx = auto_detect_pattern(args.image_dir)
    offset = min_idx - frames[0]
    px2um = args.px_to_um
    px2um2 = px2um ** 2
    first_img = None
    for f in frames:
        p = os.path.join(args.image_dir, build_filename(prefix, digs, suff, offset, f))
        if os.path.isfile(p):
            first_img = cv2.imread(p)
            break
    if first_img is None:
        raise FileNotFoundError("No sample image found")
    H, W = first_img.shape[:2]
    vw = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    cw, ch = int(0.75 * W), int(0.07 * H)
    pad_x, pad_y = int(ch * 0.1), int(ch * 0.1)
    available_text_height = ch - 2 * pad_y
    base_font_size = int(18 * (W / 1024))
    base_font_size = max(MIN_FONT_SIZE, base_font_size)
    limit_h = int(available_text_height * 0.9)
    if base_font_size > limit_h:
        base_font_size = max(MIN_FONT_SIZE, limit_h)
    area_unit = "\u00B5m\u00B2"
    for f in tqdm(frames, desc="Rendering"):
        img_p = os.path.join(args.image_dir, build_filename(prefix, digs, suff, offset, f))
        base = cv2.imread(img_p) if os.path.isfile(img_p) else np.zeros((H, W, 3), np.uint8)
        overlay = base.copy()
        areas, eccs, circs, oris = [], [], [], []
        for js, uid in grouped.get(f, []):
            pts = decode_polygon(js)
            if pts is None:
                continue
            a, e, o, c = polygon_metrics(pts, px2um2, px2um)
            if not np.isnan(a):
                areas.append(a)
            if not np.isnan(e):
                eccs.append(e)
            if not np.isnan(o):
                oris.append(o)
            if not np.isnan(c):
                circs.append(c)
            col = stable_color_for_id(uid)
            cv2.fillPoly(overlay, [pts], col)
            cv2.polylines(overlay, [pts], True, (255, 255, 255), args.contour_thick)
        blended = cv2.addWeighted(overlay, args.alpha, base, 1 - args.alpha, 0)
        mean_area = np.nanmean(areas) if areas else np.nan
        mean_ecc = np.nanmean(eccs) if eccs else np.nan
        mean_circ = np.nanmean(circs) if circs else np.nan
        mean_ori = mean_orientation_deg(oris)
        area_str = f"{mean_area:.1f} {area_unit}" if not np.isnan(mean_area) else "NA"
        ecc_str = f"{mean_ecc:.3f}" if not np.isnan(mean_ecc) else "NA"
        circ_str = f"{mean_circ:.3f}" if not np.isnan(mean_circ) else "NA"
        ori_str = f"{mean_ori:.1f}\u00B0" if not np.isnan(mean_ori) else "NA"
        frame_fmt = f"{f:0{digs}d}"
        line = (
            f"F:{frame_fmt} | N:{len(areas):03d} | "
            f"A:{area_str} | "
            f"E:{ecc_str} | "
            f"C:{circ_str} | "
            f"O:{ori_str}"
        )
        max_text_width = cw - 2 * pad_x
        out = draw_text_with_bg(
            blended, line,
            font_path=FONT_PATH,
            font_size=base_font_size,
            pos=(8, 8),
            fg=(255, 255, 255, 255),
            bg=(0, 0, 0, 140),
            padding=(pad_x, pad_y),
            max_width=max_text_width,
            max_height=ch
        )
        vw.write(out)
        cv2.imwrite(os.path.join(overlay_dir, f"overlay_{f:0{digs}d}.png"), out)
    vw.release()
    print("Done. MP4 saved at", mp4_path)

if __name__ == "__main__":
    main()
