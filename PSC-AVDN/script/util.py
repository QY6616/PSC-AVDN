from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
import base64
import csv
import json
import math
import os
import re
import sys
import cv2
import numpy as np
import requests

sys.path.append(str(SCRIPT_ROOT / "src"))
DEFAULT_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_URL = "https://www.cfgpu.com/userapi/v1/model/v1/chat/completions"
QWEN_MODEL = "qwen-vl-max-2025-01-25"
from env import compute_iou


def _corners_center(corners):
    c = np.asarray(corners, dtype=float)
    return np.mean(c, axis=0).tolist()


def data_url_from_image_array(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image for data URL")
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def clock_to_angle_deg(clock_str):
    s = str(clock_str).strip()
    m = re.fullmatch(r"(\d{1,2})(?::(\d{1,2}))?$", s)
    if not m:
        alias = {
            "forward": "12",
            "front": "12",
            "ahead": "12",
            "straight": "12",
            "right": "3",
            "left": "9",
            "back": "6",
            "backward": "6",
            "front-right": "1:30",
            "front-left": "10:30",
            "back-right": "4:30",
            "back-left": "7:30",
        }
        s = alias.get(s.lower(), "12")
        m = re.fullmatch(r"(\d{1,2})(?::(\d{1,2}))?$", s)
    hour = int(m.group(1)) % 12
    minute = int(m.group(2) or 0)
    return (hour * 30.0 + minute * 0.5) % 360.0


def generate_view_corners_with_scale(
    center_point, ob, scale_factor=1.0, angle_deg=None
):
    center_point = np.array(center_point, dtype=float).reshape(2,)
    lat_min, lng_min = ob["gps_botm_left"]
    lat_max, lng_max = ob["gps_top_right"]
    h, w = ob["map_size"][:2]
    lat_per_px = (lat_max - lat_min) / h
    lng_per_px = (lng_max - lng_min) / w
    base_pixels = 224 * float(scale_factor)
    half_lat = (base_pixels / 2) * lat_per_px
    half_lng = (base_pixels / 2) * lng_per_px
    if angle_deg is None:
        return np.array(
            [
                [center_point[0] + half_lat, center_point[1] - half_lng],
                [center_point[0] + half_lat, center_point[1] + half_lng],
                [center_point[0] - half_lat, center_point[1] + half_lng],
                [center_point[0] - half_lat, center_point[1] - half_lng],
            ],
            dtype=float,
        )
    theta = np.radians(float(angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    local = np.array(
        [
            [+half_lat, -half_lng],
            [+half_lat, +half_lng],
            [-half_lat, +half_lng],
            [-half_lat, -half_lng],
        ],
        dtype=float,
    )
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)
    rot = local @ R.T
    return rot + center_point


def create_view_image(view_corners, ob, save_path=None, out_px=768):
    try:
        map_path = os.path.join(
            ob["dataset_dir"], "train_images", f"{ob['map_name']}.tif"
        )
        if not os.path.exists(map_path):
            print(f"[WARN] Map image not found: {map_path}")
            return None
        sat_map = cv2.imread(map_path, cv2.IMREAD_COLOR)
        if sat_map is None:
            print(f"[WARN] Failed to read map image: {map_path}")
            return None
        h, w = sat_map.shape[:2]
        lat_min, lng_min = ob["gps_botm_left"]
        lat_max, lng_max = ob["gps_top_right"]
        src = []
        for lat, lng in view_corners:
            x = (lng - lng_min) / (lng_max - lng_min) * w
            y = (lat_max - lat) / (lat_max - lat_min) * h
            src.append([x, y])
        src = np.array(src, dtype=np.float32)
        src[:, 0] = np.clip(src[:, 0], 0, w - 1)
        src[:, 1] = np.clip(src[:, 1], 0, h - 1)
        if cv2.contourArea(src.astype(np.float32)) < 1.0:
            print("[WARN] View area is too small")
            return None
        out_side = int(768 if (out_px is None or out_px == "auto") else out_px)
        dst = np.array(
            [
                [0, 0],
                [out_side - 1, 0],
                [out_side - 1, out_side - 1],
                [0, out_side - 1],
            ],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(src, dst)
        patch_hr = cv2.warpPerspective(
            sat_map, M, (out_side, out_side), flags=cv2.INTER_LANCZOS4
        )
        if patch_hr is None or patch_hr.size == 0:
            print("[WARN] Perspective crop is empty")
            return None
        if save_path:
            cv2.imwrite(save_path, patch_hr, [cv2.IMWRITE_JPEG_QUALITY, 98])
        return patch_hr
    except Exception as e:
        print(f"[ERROR] Crop failed: {e}")
        return None


def save_traj_boxes_debug_image(
    tif_dataset_dir, map_name, ob, predicted_boxes, out_path, zoom_highlight_idx=None
):
    import os, cv2, numpy as np

    try:
        map_path = os.path.join(tif_dataset_dir, f"{map_name}.tif")
        if not os.path.exists(map_path):
            print(f"[WARN] Map image not found: {map_path}")
            return
        img = cv2.imread(map_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read map image: {map_path}")
            return
        h, w = img.shape[:2]
        lat_min, lng_min = ob["gps_botm_left"]
        lat_max, lng_max = ob["gps_top_right"]

        def gps_to_img_coords_f(latlng):
            lat, lng = float(latlng[0]), float(latlng[1])
            x = (lng - lng_min) / (lng_max - lng_min) * w
            y = (lat_max - lat) / (lat_max - lat_min) * h
            return np.array([x, y], dtype=float)

        def gps_quad_to_px_f(corners_ll):
            return np.array(
                [gps_to_img_coords_f(corners_ll[k]) for k in range(4)], dtype=float
            )

        def draw_aabb_from_poly(poly_px_f, color, thickness):
            xs = poly_px_f[:, 0]
            ys = poly_px_f[:, 1]
            x1 = int(np.floor(xs.min()))
            y1 = int(np.floor(ys.min()))
            x2 = int(np.ceil(xs.max()))
            y2 = int(np.ceil(ys.max()))
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        gt_list = ob.get("gt_path_corners", [])
        if isinstance(gt_list, (list, tuple)) and len(gt_list) >= 1:
            start_c = np.array(gt_list[0], dtype=float)
            poly_s_f = gps_quad_to_px_f(start_c)
            poly_s = np.round(poly_s_f).astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(img, [poly_s], -1, (0, 255, 0), 2)
            cs_px = np.round(poly_s_f.mean(axis=0)).astype(int)
            cv2.putText(
                img,
                "GT-S",
                (cs_px[0] + 4, cs_px[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                "GT-S",
                (cs_px[0] + 4, cs_px[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        if isinstance(gt_list, (list, tuple)) and len(gt_list) >= 2:
            end_c = np.array(gt_list[-1], dtype=float)
            poly_e_f = gps_quad_to_px_f(end_c)
            poly_e = np.round(poly_e_f).astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(img, [poly_e], -1, (0, 0, 255), 2)
            ce_px = np.round(poly_e_f.mean(axis=0)).astype(int)
            cv2.putText(
                img,
                "GT-E",
                (ce_px[0] + 4, ce_px[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                "GT-E",
                (ce_px[0] + 4, ce_px[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        centers_f = []
        N = len(predicted_boxes)
        for idx, corners_ll in enumerate(predicted_boxes):
            corners_ll = np.array(corners_ll, dtype=float)
            poly_px_f = gps_quad_to_px_f(corners_ll)
            poly_px = np.round(poly_px_f).astype(np.int32).reshape(-1, 1, 2)
            thick = (
                3
                if (zoom_highlight_idx is not None and idx == zoom_highlight_idx)
                else 1
            )
            is_last = idx == N - 1
            is_zoom = zoom_highlight_idx is not None and idx == zoom_highlight_idx
            if is_zoom or (is_last and zoom_highlight_idx is None):
                draw_aabb_from_poly(poly_px_f, (255, 255, 255), thick)
            else:
                cv2.drawContours(img, [poly_px], -1, (255, 255, 255), thick)
            c_px_f = poly_px_f.mean(axis=0)
            centers_f.append(c_px_f)
            r = max(3, int(0.006 * max(h, w)))
            cv2.circle(
                img,
                (int(round(c_px_f[0])), int(round(c_px_f[1]))),
                r,
                (0, 0, 255),
                thickness=-1,
            )
            label_main = f"P{idx}"
            cv2.putText(
                img,
                label_main,
                (int(round(c_px_f[0])) + 4, int(round(c_px_f[1])) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                label_main,
                (int(round(c_px_f[0])) + 4, int(round(c_px_f[1])) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if idx > 0:
                p_prev = centers_f[idx - 1]
                cv2.line(
                    img,
                    (int(round(p_prev[0])), int(round(p_prev[1]))),
                    (int(round(c_px_f[0])), int(round(c_px_f[1]))),
                    (0, 255, 255),
                    2,
                )
        try:
            if len(predicted_boxes) >= 2 and len(centers_f) >= 2:
                pb1 = np.array(predicted_boxes[1], dtype=float)
                poly1_px_f = gps_quad_to_px_f(pb1)
                top_mid_px = 0.5 * (poly1_px_f[0] + poly1_px_f[1])
                bot_mid_px = 0.5 * (poly1_px_f[3] + poly1_px_f[2])
                hv = top_mid_px - bot_mid_px
                hv_norm = float(np.linalg.norm(hv))
                if hv_norm < 1e-6:
                    hv = centers_f[1] - centers_f[0]
                    hv_norm = float(np.linalg.norm(hv))
                if hv_norm >= 1e-6:
                    w1 = np.linalg.norm(poly1_px_f[1] - poly1_px_f[0])
                    h1 = np.linalg.norm(poly1_px_f[2] - poly1_px_f[1])
                    L = max(12.0, 0.6 * float(min(w1, h1)))
                    dir_vec = hv / hv_norm
                    p0 = centers_f[0]
                    p1 = p0 + dir_vec * L
                    cv2.arrowedLine(
                        img,
                        (int(round(p0[0])), int(round(p0[1]))),
                        (int(round(p1[0])), int(round(p1[1]))),
                        (255, 0, 0),
                        2,
                        tipLength=0.25,
                    )
        except Exception as _e:
            print(f"[WARN] Failed to draw the first-step turn arrow:{_e}")
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    except Exception as e:
        print(f"[WARN] Failed to save trajectory-box debug image: {e}")


CLOCK_TABLE = [
    "12",
    "12:30",
    "1",
    "1:30",
    "2",
    "2:30",
    "3",
    "3:30",
    "4",
    "4:30",
    "5",
    "5:30",
    "6",
    "6:30",
    "7",
    "7:30",
    "8",
    "8:30",
    "9",
    "9:30",
    "10",
    "10:30",
    "11",
    "11:30",
]


def angle_to_clock(rel_deg: float) -> str:
    rel = (float(rel_deg) % 360.0 + 360.0) % 360.0
    idx = int(round(rel / 15.0)) % 24
    return CLOCK_TABLE[idx]


SECTOR2DEG = {
    "north": 0,
    "n": 0,
    "northeast": 45,
    "ne": 45,
    "east": 90,
    "e": 90,
    "southeast": 135,
    "se": 135,
    "south": 180,
    "s": 180,
    "southwest": 225,
    "sw": 225,
    "west": 270,
    "w": 270,
    "northwest": 315,
    "nw": 315,
}


def _extract_content(resp):
    try:
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            return "".join(
                [c.get("text", "") if isinstance(c, dict) else str(c) for c in content]
            )
        return content or ""
    except Exception:
        return resp.get("output_text") or resp.get("content") or ""


def _parse_json_loose(raw: str):
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _norm_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        x = x.strip().lower()
        return x in ("true", "yes", "y", "1")
    return bool(x)


def _pixel_to_latlng_with_corners(px, py, corners_latlng, img_w, img_h):
    world_pts = np.array(
        [
            [corners_latlng[0][1], corners_latlng[0][0]],
            [corners_latlng[1][1], corners_latlng[1][0]],
            [corners_latlng[2][1], corners_latlng[2][0]],
            [corners_latlng[3][1], corners_latlng[3][0]],
        ],
        dtype=np.float32,
    )
    img_pts = np.array(
        [[0, 0], [img_w - 1, 0], [img_w - 1, img_h - 1], [0, img_h - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(img_pts, world_pts)
    v = np.array([px, py, 1.0], dtype=np.float32)
    w = H @ v
    if abs(w[2]) < 1e-8:
        lat_c = float(np.mean([c[0] for c in corners_latlng]))
        lng_c = float(np.mean([c[1] for c in corners_latlng]))
        return [lat_c, lng_c]
    w = w / w[2]
    lng, lat = float(w[0]), float(w[1])
    return [lat, lng]


def _move_forward_geo(pos_latlng, heading_deg, meters=40.0):
    lat, lng = float(pos_latlng[0]), float(pos_latlng[1])
    theta = math.radians(heading_deg % 360.0)
    dn = meters * math.cos(theta)
    de = meters * math.sin(theta)
    dlat = dn / 111320.0
    denom = 111320.0 * max(1e-6, math.cos(math.radians(lat)))
    dlng = de / denom
    return [lat + dlat, lng + dlng]


def decide_turn_from_land(
    instruction,
    patch_bgr,
    api_key=DEFAULT_API_KEY,
    url=QWEN_URL,
    model=QWEN_MODEL,
    timeout=120,
    *,
    view="absolute",
    heading_deg=None,
):
    try:
        data_url = data_url_from_image_array(patch_bgr)
        if view == "absolute":
            system_prompt = (
                "You are a drone navigation assistant. You are given an aerial image (the top of the image represents true north)."
                "Based on the user's instruction, identify the absolute direction of the destination and strictly return it in JSON format:"
                '{"abs_deg":0-359, "abs_dir":"N|NE|E|SE|S|SW|W|NW|or Chinese directional text", '
                '"confidence":0.0-1.0, "reason":"At least 50 characters, no more than 120 characters, key clues"}'
                "Do not output clock positions; we will calculate them based on the current heading angle."
                "The 'reason' field must not be empty, even if the target cannot be identified."
                "Your focus is to determine the absolute direction of the destination based on location-based directional information in the instruction."
            )
            heading_text = f"Current heading (relative to true north, clockwise): {(heading_deg if heading_deg is not None else 0):.1f} degrees."
        else:
            system_prompt = (
                "You are a drone navigation assistant. You are given an aerial image (the top of the image represents the current heading direction)."
                "Based on the user's instruction, identify the [relative angle to the current heading] and strictly return it in JSON format:"
                '{"rel_deg":0-359, "confidence":0.0-1.0, "reason":"No more than 120 characters, at least 50 characters"}'
                "The 'reason' field must not be empty, even if the target cannot be identified."
            )
            heading_text = ""
        user_text = f"Instruction: {instruction}\n{heading_text}Return only in JSON format, without any additional text."
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": model,
            "stream": False,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        ct = (r.headers.get("Content-Type") or "").lower()
        if "application/json" not in ct:
            return "12", "fallback: non-json response", {"raw": r.text[:200]}
        try:
            resp = r.json()
        except Exception as je:
            return "12", f"fallback: json parse error ({je})", {"raw": r.text[:200]}
        if isinstance(resp, dict) and "error" in resp:
            return "12", f"fallback: api error {resp.get('error')}", {"raw": resp}
        raw = _extract_content(resp).strip()
        abs_deg = None
        abs_dir = None
        rel_deg = None
        reason = ""
        confidence = None
        parsed = _parse_json_loose(raw)
        if parsed is not None:
            abs_deg = parsed.get("abs_deg")
            abs_dir = parsed.get("abs_dir")
            rel_deg = parsed.get("rel_deg")
            reason = (parsed.get("reason") or parsed.get("explanation") or "").strip()
            confidence = parsed.get("confidence")
        if view == "absolute":
            if abs_deg is None and abs_dir:
                key = str(abs_dir).strip().lower()
                abs_deg = SECTOR2DEG.get(abs_dir, SECTOR2DEG.get(key))
            if abs_deg is None:
                m = re.search(r"(\d{1,3})\s*°", raw)
                if m:
                    abs_deg = int(m.group(1)) % 360
                else:
                    for k, v in SECTOR2DEG.items():
                        if k in raw.lower() or k in raw:
                            abs_deg = v
                            break
            if abs_deg is None:
                clock = "12"
                meta = {
                    "abs_deg": None,
                    "rel_deg": 0.0,
                    "heading_deg": heading_deg,
                    "confidence": confidence,
                    "raw": raw,
                }
                return clock, (reason or "N/A"), meta
            rel_deg = (float(abs_deg) - float(heading_deg or 0.0)) % 360.0
        else:
            if rel_deg is None:
                m = re.search(r"(\d{1,3})\s*°", raw)
                rel_deg = float(m.group(1)) % 360 if m else 0.0
        clock = angle_to_clock(rel_deg)
        meta = {
            "abs_deg": (abs_deg if view == "absolute" else None),
            "rel_deg": float(rel_deg),
            "heading_deg": float(heading_deg or 0.0),
            "confidence": confidence,
            "raw": raw,
        }
        return clock, (reason or "N/A"), meta
    except Exception as e:
        return "12", f"fallback: exception {e}", {}


MAP_CACHE = {}


def _get_map_image_cached(ob):
    map_path = os.path.join(ob["dataset_dir"], "train_images", f"{ob['map_name']}.tif")
    img = MAP_CACHE.get(map_path)
    if img is not None:
        return img
    if not os.path.exists(map_path):
        print(f"[WARN] Map image not found: {map_path}")
        return None
    img = cv2.imread(map_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Failed to read map image: {map_path}")
        return None
    MAP_CACHE[map_path] = img
    return img


def _gps_to_img_px(latlng, ob, img_w, img_h):
    lat_min, lng_min = ob["gps_botm_left"]
    lat_max, lng_max = ob["gps_top_right"]
    lat, lng = float(latlng[0]), float(latlng[1])
    x = (lng - lng_min) / (lng_max - lng_min) * img_w
    y = (lat_max - lat) / (lat_max - lat_min) * img_h
    return float(np.clip(x, 0, img_w - 1)), float(np.clip(y, 0, img_h - 1))


def draw_pos_on_patch(patch_bgr, corners_ll, pos_latlng=None, color=(0, 0, 255)):
    import numpy as np, cv2

    h, w = patch_bgr.shape[:2]
    if pos_latlng is None:
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    else:
        ll = np.asarray(corners_ll, np.float32)
        lat0 = float(ll[:, 0].mean())
        lon0 = float(ll[:, 1].mean())
        kx = 111320.0 * np.cos(np.deg2rad(lat0))
        ky = 110540.0

        def ll2xy(a):
            a = np.asarray(a, np.float32)
            x = (a[:, 1] - lon0) * kx
            y = -(a[:, 0] - lat0) * ky
            return np.stack([x, y], 1)

        src = ll2xy(ll)
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        p_xy = ll2xy(
            np.array([[float(pos_latlng[0]), float(pos_latlng[1])]], np.float32)
        )
        ph = np.hstack([p_xy, np.ones((1, 1), np.float32)])
        uvw = (H @ ph.T).ravel()
        cx, cy = float(uvw[0] / max(uvw[2], 1e-6)), float(uvw[1] / max(uvw[2], 1e-6))
    r = max(3, int(0.006 * max(h, w)))
    cv2.circle(patch_bgr, (int(round(cx)), int(round(cy))), r, color, thickness=-1)
    cv2.circle(patch_bgr, (int(round(cx)), int(round(cy))), r + 1, (255, 255, 255), 1)
    return patch_bgr


def make_minimap_crop_basemap(
    ob,
    corners_seq,
    img_seq=None,
    out_path=None,
    pad_m=30.0,
    long_side_px=768,
    jpeg_quality=85,
    save_webp=False,
    force_size=(768, 768),
    pad_color=(240, 240, 240),
):
    if not corners_seq:
        return None
    map_path = os.path.join(ob["dataset_dir"], "train_images", f"{ob['map_name']}.tif")
    sat_map = cv2.imread(map_path, cv2.IMREAD_COLOR)
    if sat_map is None:
        W = force_size[0] if (force_size is not None) else int(long_side_px or 768)
        H = force_size[1] if (force_size is not None) else int(long_side_px or 768)

        def make_graywhite_mosaic(h, w, tile=12, c0=220, c1=245):
            bg = np.empty((h, w, 3), np.uint8)
            ys = np.arange(h) // tile
            xs = np.arange(w) // tile
            grid = (ys[:, None] + xs[None, :]) % 2
            bg[:] = c0
            bg[grid == 1] = c1
            return bg

        fallback = make_graywhite_mosaic(int(H), int(W))
        if out_path:
            _save_img_any(fallback, out_path, jpeg_quality, save_webp)
        return fallback
    H_img, W_img = sat_map.shape[:2]
    lat_min, lng_min = ob["gps_botm_left"]
    lat_max, lng_max = ob["gps_top_right"]

    def ll_to_img_xy(lat, lng):
        X = (lng - lng_min) / (lng_max - lng_min) * W_img
        Y = (lat_max - lat) / (lat_max - lat_min) * H_img
        return np.array([X, Y], np.float32)

    quads_px = [
        np.array([ll_to_img_xy(pt[0], pt[1]) for pt in c4], np.float32)
        for c4 in corners_seq
    ]
    centers_px = np.array([qp.mean(0) for qp in quads_px], np.float32)
    px_per_deg_x = W_img / float((lng_max - lng_min) + 1e-12)
    px_per_deg_y = H_img / float((lat_max - lat_min) + 1e-12)
    all_ll = np.asarray(corners_seq, float).reshape(-1, 2)
    lat0 = float(all_ll[:, 0].mean())
    kx = 111320.0 * np.cos(np.deg2rad(lat0))
    ky = 110540.0
    px_per_m_x = px_per_deg_x / max(1e-12, kx)
    px_per_m_y = px_per_deg_y / max(1e-12, ky)
    px_per_m = float(0.5 * (px_per_m_x + px_per_m_y))
    px_pad = int(max(4, round(pad_m * px_per_m)))
    all_pts = np.vstack(quads_px)
    min_xy = np.floor(all_pts.min(0) - px_pad).astype(int)
    max_xy = np.ceil(all_pts.max(0) + px_pad).astype(int)
    min_x = int(np.clip(min_xy[0], 0, W_img - 1))
    min_y = int(np.clip(min_xy[1], 0, H_img - 1))
    max_x = int(np.clip(max_xy[0], min_x + 1, W_img))
    max_y = int(np.clip(max_xy[1], min_y + 1, H_img))
    crop = sat_map[min_y:max_y, min_x:max_x].copy()
    h, w = crop.shape[:2]
    off = np.array([min_x, min_y], np.float32)
    quads_c = [qp - off for qp in quads_px]
    centers_c = centers_px - off
    if long_side_px and max(w, h) > long_side_px:
        s = float(long_side_px) / max(w, h)
        new_w, new_h = int(round(w * s)), int(round(h * s))
        crop = cv2.resize(crop, (new_w, new_h), cv2.INTER_AREA)
        scale = np.array([s, s], np.float32)
        quads_c = [qp * scale for qp in quads_c]
        centers_c = centers_c * scale
        h, w = crop.shape[:2]

    def make_graywhite_mosaic(h, w, tile=12, c0=220, c1=245):
        bg = np.empty((h, w, 3), np.uint8)
        ys = np.arange(h) // tile
        xs = np.arange(w) // tile
        grid = (ys[:, None] + xs[None, :]) % 2
        bg[:] = c0
        bg[grid == 1] = c1
        return bg

    mosaic_bg = make_graywhite_mosaic(h, w)
    mask = np.zeros((h, w), np.uint8)
    polys_pts = [np.floor(qp + 0.5).astype(np.int32).reshape(-1, 2) for qp in quads_c]
    if len(polys_pts) > 0:
        pts_all = np.vstack(polys_pts).astype(np.int32)
        hull = cv2.convexHull(pts_all)
        cv2.fillConvexPoly(mask, hull, 255, lineType=cv2.LINE_AA)
    out = mosaic_bg.copy()
    crop_fg = cv2.bitwise_and(crop, crop, mask=mask)
    out_bg = cv2.bitwise_and(out, out, mask=cv2.bitwise_not(mask))
    out = cv2.add(out_bg, crop_fg)
    aa = cv2.LINE_AA
    base = float(max(h, w))
    thick = max(1, int(round(0.0030 * base)))
    r_dot = max(2, int(round(0.0060 * base)))
    r_outline = max(1, int(round(0.0018 * base)))
    tip_len = 0.25
    min_arrow = max(8, int(round(0.010 * base)))
    max_arrow = int(round(0.060 * base))
    col_line = (0, 255, 255)
    col_arrow = (0, 0, 255)
    col_dot = (0, 0, 255)
    col_edge = (255, 255, 255)
    ctrs = np.floor(centers_c + 0.5).astype(np.int32)
    if len(ctrs) >= 1:
        if len(ctrs) >= 2:
            pts_poly = ctrs.reshape(-1, 1, 2)
            cv2.polylines(
                out, [pts_poly], False, col_line, thickness=thick, lineType=aa
            )
        for i in range(len(ctrs)):
            p = tuple(ctrs[i])
            cv2.circle(out, p, r_dot + r_outline, col_edge, -1, lineType=aa)
            cv2.circle(out, p, r_dot, col_dot, -1, lineType=aa)
            if i + 1 < len(ctrs):
                q = ctrs[i + 1]
                vec = np.array([q[0] - p[0], q[1] - p[1]], np.float32)
                d = float(np.linalg.norm(vec))
                if d > 1e-6:
                    vhat = vec / d
                    start = np.array(p, np.float32) + vhat * (r_dot + r_outline + 1)
                    head_len = max(10, int(round(0.035 * base)))
                    head_wid = max(8, int(round(0.022 * base)))
                    end_f = np.array([q[0], q[1]], np.float32)
                    base_center = end_f - vhat * head_len
                    cv2.line(
                        out,
                        (int(round(start[0])), int(round(start[1]))),
                        (int(round(base_center[0])), int(round(base_center[1]))),
                        col_line,
                        max(1, thick),
                        lineType=aa,
                    )
                    n_hat = np.array([-vhat[1], vhat[0]], np.float32)
                    tip = (int(round(end_f[0])), int(round(end_f[1])))
                    p2 = base_center + n_hat * (head_wid * 0.5)
                    p3 = base_center - n_hat * (head_wid * 0.5)
                    tri = np.array(
                        [
                            [tip[0], tip[1]],
                            [int(round(p2[0])), int(round(p2[1]))],
                            [int(round(p3[0])), int(round(p3[1]))],
                        ],
                        np.int32,
                    )
                    cv2.fillConvexPoly(out, tri, col_arrow, lineType=aa)
    if force_size is not None:
        tgt_w, tgt_h = int(force_size[0]), int(force_size[1])
        cur_h, cur_w = out.shape[:2]
        s = min(tgt_w / float(cur_w), tgt_h / float(cur_h))
        new_w = max(1, int(round(cur_w * s)))
        new_h = max(1, int(round(cur_h * s)))
        if new_w != cur_w or new_h != cur_h:
            out = cv2.resize(out, (new_w, new_h), cv2.INTER_AREA)
        canvas = make_graywhite_mosaic(tgt_h, tgt_w)
        off_x = (tgt_w - new_w) // 2
        off_y = (tgt_h - new_h) // 2
        canvas[off_y : off_y + new_h, off_x : off_x + new_w] = out
        out = canvas
    if out_path:
        _save_img_any(out, out_path, jpeg_quality, save_webp)
    return out


def _save_img_any(img, out_path, jpeg_quality=85, save_webp=False):
    ext = (out_path.split(".")[-1] or "").lower()
    if save_webp or ext == "webp":
        cv2.imwrite(
            out_path if ext == "webp" else (out_path.rsplit(".", 1)[0] + ".webp"),
            img,
            [cv2.IMWRITE_WEBP_QUALITY, int(max(50, min(100, jpeg_quality)))],
        )
    else:
        cv2.imwrite(
            out_path,
            img,
            [cv2.IMWRITE_JPEG_QUALITY, int(max(50, min(95, jpeg_quality)))],
        )
