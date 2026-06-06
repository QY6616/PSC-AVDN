import os
import sys
import re
import csv
import json
import math
import base64
import requests
import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import ANDHNavBatch
from torch.utils.data import DataLoader
from util import (
    generate_view_corners_with_scale,
    create_view_image,
    decide_turn_from_land,
    draw_pos_on_patch,
    _corners_center,
    angle_to_clock,
    clock_to_angle_deg,
    save_traj_boxes_debug_image,
    compute_iou,
    make_minimap_crop_basemap,
    _norm_bool,
    _move_forward_geo,
    _extract_content,
    _parse_json_loose,
    data_url_from_image_array
)

RESULTS_CSV = str(PROJECT_ROOT / "preds_out" / "parsing_results.csv")

ANNO_DIR = str(PROJECT_ROOT / "datasets" / "AVDN" / "annotations")
DATASET_DIR = str(PROJECT_ROOT / "datasets" / "AVDN")
SPLIT       = "test_unseen"
PRED_DIR = str(PROJECT_ROOT / "preds" / "andh")
OUT_DIR     = os.path.join(PRED_DIR, "search_output")
SCALE_FACTOR       = 5
FIXED_CROP_SIDE    = 768

DEFAULT_API_KEY = os.getenv("API_KEY", "")
QWEN_URL        = ""
QWEN_MODEL      = ""

def _haversine_m(lat1, lng1, lat2, lng2):
    R = 6371000.0
    phi1 = math.radians(lat1);  phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb/2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return R * c

def _patch_xy_to_latlng(u, v, view_corners, ob):
    map_path = os.path.join(ob['dataset_dir'], 'train_images', f"{ob['map_name']}.tif")
    sat_map = cv2.imread(map_path, cv2.IMREAD_COLOR)
    if sat_map is None:
        raise RuntimeError(f"read map fail: {map_path}")
    H_img, W_img = sat_map.shape[:2]
    lat_min, lng_min = ob['gps_botm_left']
    lat_max, lng_max = ob['gps_top_right']

    src = []
    for lat, lng in view_corners:
        X = (lng - lng_min) / (lng_max - lng_min) * W_img
        Y = (lat_max - lat) / (lat_max - lat_min) * H_img
        src.append([X, Y])
    src = np.array(src, dtype=np.float32)
    src[:, 0] = np.clip(src[:, 0], 0, W_img - 1)
    src[:, 1] = np.clip(src[:, 1], 0, H_img - 1)

    side = int(768)
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype=np.float32)

    H_inv = cv2.getPerspectiveTransform(dst, src)

    vec = H_inv @ np.array([float(u), float(v), 1.0], dtype=np.float32)
    X = float(vec[0] / (vec[2] + 1e-12))
    Y = float(vec[1] / (vec[2] + 1e-12))

    lng = lng_min + (X / W_img) * (lng_max - lng_min)
    lat = lat_max - (Y / H_img) * (lat_max - lat_min)
    return [lat, lng]

def qwen_locate_bbox_in_view(
    dest_desc,
    patch_main_bgr,
    patch_s4_bgr=None,
    patch_s8_bgr=None,
    minimap_bgr=None,
    api_key=DEFAULT_API_KEY,
    url=QWEN_URL,
    model=QWEN_MODEL,
    timeout=120,
    restrict_top_half=True,
    prev_grid_5x5=None
):
    if restrict_top_half is True:
        focus_text = (
            f"[Task Decomposition / Thought Process]\n"
            f"\n"
            f"Views:\n"
            f"- Image 1: main view ({FIXED_CROP_SIDE}x{FIXED_CROP_SIDE}), top = current heading. THIS IS THE ONLY IMAGE YOU MAY OUTPUT A BOX FOR.\n"
            f"- Image 2: same center and heading as Image 1 but with a narrower, more zoomed-in view of the same moment (local detail context).\n"
            f"- Image 3: same center and heading as Image 1 but with a wider, more zoomed-out view of the same moment (broader surrounding context).\n"
            f"- Image 4: north-up global map; red dot = current position, red arrow = current heading, "
            f"yellow trail = recent trajectory.\n"
            f"Note: Image 2 and Image 3 show exactly the same location and heading as Image 1, only with different field-of-view sizes. "
            f"Use them to confirm ambiguous regions in Image 1. For example, if the target in Image 1 is blurry, unclear, "
            f"or lies near the border of Image 1, check Image 2 / Image 3 to verify what that region actually is before deciding.\n"
            f"\n"
            f"1) Destination parsing:\n"
            f"   - Parse the destination description and extract key visual/physical attributes "
            f"(shape, texture, material, color), geometric structure, and any directional / positional cues.\n"
            f"\n"
            f"2) Scene understanding:\n"
            f"   - Summarize the scene and spatial layout in Image 1 (main view).\n"
            f"   - Summarize local spatial / contextual structure seen in Image 2 and Image 3 "
            f"(they provide tighter or wider context of the exact same place and heading).\n"
            f"   - Separately summarize global position / heading / trajectory cues from Image 4 (the minimap).\n"
            f"\n"
            f"3)Generate semantic grid map on Image 1:\n"
            f"   - Partition Image 1 into 5 rows × 5 columns (top→bottom, left→right).\n"
            f"   - For EACH cell, output a label chosen ONLY from this closed set (lowercase; 1–2 words): road, building, container yard, parking, runway, water, field, forest, roof, shadow, rail, vehicle; "
            f"use 'unknown' if unclear.\n"
            f"   - Return this grid in JSON under key \"grid_5x5\" as a 5×5 array of strings.\n"
            f"\n"
            f"4) Target localization ON IMAGE 1 ONLY:\n"
            f"   - First, restrict your search to the TOP HALF of Image 1 (forward / current heading direction).\n"
            f"   - Propose candidate regions in Image 1 that satisfy the destination description, using appearance, "
            f"geometry, texture, shadows, arrangement, and nearby structures.\n"
            f"   - If nothing in the top half matches, you may extend to the full Image 1, "
            f"but the final chosen region must still come from Image 1.\n"
            f"   - Choose EXACTLY ONE final region. If you cannot isolate one unambiguous region in Image 1, "
            f"you must report dest_present=false.\n"
            f"\n"
            f"5) Bounding box output:\n"
            f"   - Produce the minimal square bounding box that fully encloses the chosen region ON IMAGE 1 ONLY.\n"
            f"   - All bounding box coordinates MUST be given in Image 1's {FIXED_CROP_SIDE}x{FIXED_CROP_SIDE} pixel grid.\n"
            f"   - Return dest_present (true/false), bbox_2d, confidence in [0.0,1.0], "
            f"and a brief reason (<500 chars).\n"
            f"\n"
            f"Rules:\n"
            f"- NEVER output or reference a bbox on Image 2, Image 3, or Image 4; they are context only.\n"
            f"- If you are not sufficiently confident in ONE clear match in Image 1, output dest_present=false.\n"
            f"- These are overhead / satellite views; color contrast may be weak. "
            f"Rely on shape, texture, layout, shadows, and stable spatial relations, not just raw color."
        )
    else:
        focus_text = (
            f"[Decision Mode and Localization Procedure]\n"
            f"\n"
            f"Views:\n"
            f"- Image 1: main view ({FIXED_CROP_SIDE}x{FIXED_CROP_SIDE}), top = current heading. THIS IS THE ONLY IMAGE YOU MAY OUTPUT A BOX FOR.\n"
            f"- Image 2: same center and heading as Image 1 but with a narrower, more zoomed-in view of the same moment (local detail context).\n"
            f"- Image 3: same center and heading as Image 1 but with a wider, more zoomed-out view of the same moment (broader surrounding context).\n"
            f"- Image 4: north-up global map; red dot = current position, red arrow = current heading, "
            f"yellow trail = recent trajectory.\n"
            f"Note: Image 2 and Image 3 show exactly the same location and heading as Image 1, only with different field-of-view sizes. "
            f"Use them to disambiguate candidates in Image 1, especially if a region is small, noisy, or sits at the border of Image 1.\n"
            f"\n"
            f"Step 0 - Mode Selection (Simple Direct Localization vs. Reasoning Chain):\n"
            f"- If the destination description is just a single category / attribute phrase "
            f"(only color / material / shape / category, e.g. 'a white house', 'a circular water tower'), "
            f"and it does NOT include relational / compositional constraints "
            f"(left/right/center/adjacent/along/between/leftmost/rightmost/near/connected to/parallel to), "
            f"then use [Simple Direct Localization].\n"
            f"- Otherwise, use [Reasoning Chain].\n"
            f"\n"
            f"[Simple Direct Localization]:\n"
            f"- Work directly on Image 1. Start from the TOP HALF of Image 1 (the forward / heading direction), "
            f"expanding to the full Image 1 only if necessary.\n"
            f"- Find the best-matching region in Image 1 using category- and attribute-level cues "
            f"(roof shape, texture pattern, shadow geometry, surroundings, etc.).\n"
            f"- If multiple candidates in Image 1 look similar, consult Image 2 and Image 3 "
            f"to clarify local spatial context (e.g. cluster layout, adjacency to a runway / road), "
            f"and consult Image 4 (the minimap) for global orientation / trajectory. "
            f"Use this context to decide which ONE candidate in Image 1 is most consistent.\n"
            f"\n"
            f"[Reasoning Chain]:\n"
            f"Step 1 - Generate an explicit reasoning chain. Break the destination description into ordered, "
            f"checkable sub-steps. Example: 'the vehicle on the far left of the central parking line' → "
            f"'identify the central parking line' → 'scan along that line for the leftmost vehicle'.\n"
            f"Step 2 - Follow those sub-steps on Image 1. "
            f"For each sub-step, progressively narrow the candidate area in Image 1, "
            f"starting with the top half first whenever it applies. "
            f"Use Image 2 / Image 3 only to clarify local geometry and layout "
            f"(e.g. which elongated strip is a runway, which dense blob is a parking lot), "
            f"and use Image 4 only for high-level position / heading / trajectory context. "
            f"Record at least one concrete visual evidence item for each critical sub-step.\n"
            f"\n"
            f"[Generate semantic grid map on Image 1]:\n"
            f"   - Partition Image 1 into 5 rows × 5 columns (top→bottom, left→right).\n"
            f"   - For EACH cell, output a label chosen ONLY from this closed set (lowercase; 1–2 words): road, building, container yard, parking, runway, water, field, forest, roof, shadow, rail, vehicle; "
            f"use 'unknown' if unclear.\n"
            f"   - Return this grid in JSON under key \"grid_5x5\" as a 5×5 array of strings.\n"
            f"\n"
            f"[Bounding box output]:\n"
            f"- Draw one tight square bounding box that encloses the final chosen region ON IMAGE 1 ONLY.\n"
            f"- All bbox coordinates MUST be reported in Image 1's {FIXED_CROP_SIDE}x{FIXED_CROP_SIDE} pixel grid.\n"
            f"- Return dest_present (true/false), bbox_2d, confidence in [0.0,1.0], and 1–3 short justification sentences.\n"
            f"  · In Simple Direct Localization mode: name the decisive visual attributes.\n"
            f"  · In Reasoning Chain mode: cite the key reasoning steps and supporting evidence.\n"
            f"\n"
            f"Critical rules:\n"
            f"- DO NOT output bounding boxes for Image 2, Image 3, or Image 4; they are context only.\n"
            f"- If you cannot isolate exactly one high-confidence match within Image 1, "
            f"you MUST output dest_present=false.\n"
            f"- These are overhead / satellite views; color may be low-contrast. "
            f"Use structure, texture, shape, arrangement, shadows, and spatial relations, not just raw color."
        )

    system_prompt = (
        "You are a visual localization assistant. You must return ONLY strict JSON, with no extra text.\n"
        "\n"
        "You will receive up to 4 images:\n"
        f"- Image 1: main view ({FIXED_CROP_SIDE}x{FIXED_CROP_SIDE}), top = current heading. "
        "This is the ONLY image for which you are allowed to output a bounding box.\n"
        "- Image 2: same center and same heading as Image 1 but with a narrower, more zoomed-in view of the exact same moment. "
        "Use this only as local-detail context.\n"
        "- Image 3: same center and same heading as Image 1 but with a wider, more zoomed-out view of the exact same moment. "
        "Use this only as surrounding-context.\n"
        "- Image 4: a north-up global map. A red solid dot marks the current position, "
        "a red arrow marks the current heading, and a yellow trail shows the recent trajectory. "
        "Use this only as high-level positional / directional prior.\n"
        "\n"
        "If a previous-step grid_5x5 is provided:\n"
        "- Treat it as a SOFT PRIOR for Image 1 ONLY (top→bottom rows, left→right cols).\n"
        "- Use it just to prioritize scanning cells and keep temporal consistency.\n"
        "- If it conflicts with current evidence in Image 1/2/3, IGNORE the prior.\n"
        "- Always regenerate the current step's grid_5x5 from the present images; do NOT copy the prior.\n"
        "\n"
        "IMPORTANT:\n"
        "- Image 2, Image 3, and Image 4 are context only. NEVER output a bbox for them.\n"
        "- All bbox coordinates MUST be reported in Image 1's pixel grid "
        f"({FIXED_CROP_SIDE}x{FIXED_CROP_SIDE}).\n"
        "- If you cannot confidently identify exactly one valid region in Image 1, "
        "you must set dest_present=false.\n"
        "\n"
        "Your task is described below. Follow it exactly.\n"
        f"{focus_text}\n"
        "\n"
        "You must output ONLY the following JSON object:\n"
        "{"
        "\"dest_present\": true|false, "
        "\"bbox_2d\": [x1,y1,x2,y2], "
        "\"confidence\": 0.0-1.0, "
        "\"reason\": \"a justification of up to 500 characters\", "
        "\"grid_5x5\": [[c11,c12,c13,c14,c15],[c21,c22,c23,c24,c25],[c31,c32,c33,c34,c35],[c41,c42,c43,c44,c45],[c51,c52,c53,c54,c55]]"
        "}"
    )

    user_msg = (
        f"Destination description: {dest_desc or '(None)'}\n"
        f"All coordinates MUST be in Image 1's {FIXED_CROP_SIDE}x{FIXED_CROP_SIDE} pixel grid.\n"
        f"If not sure, return dest_present=false."
    )

    content = [
        {"type": "text", "text": user_msg},

        {"type": "image_url", "image_url": {
            "url": data_url_from_image_array(patch_main_bgr)
        }},
    ]

    if prev_grid_5x5:
        content.append({
            "type": "text",
            "text": "Previous step grid_5x5 (Image 1, soft prior; ignore if inconsistent): " + json.dumps(prev_grid_5x5)
        })

    if patch_s4_bgr is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url_from_image_array(patch_s4_bgr)}
        })
    if patch_s8_bgr is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url_from_image_array(patch_s8_bgr)}
        })

    if minimap_bgr is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url_from_image_array(minimap_bgr)}
        })

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model": model,
        "temperature": 0,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content}
        ]
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        ct = (r.headers.get("Content-Type") or "").lower()
        if "application/json" not in ct:
            return {
                "dest_present": False,
                "bbox_2d": [0,0,0,0],
                "confidence": 0.0,
                "reason": "fallback: non-json response",
                "grid_5x5": [],
                "raw": r.text[:200]
            }
        resp = r.json()
    except Exception as e:
        return {
            "dest_present": False,
            "bbox_2d": [0,0,0,0],
            "confidence": 0.0,
            "reason": f"fallback: request/json error: {e}",
            "grid_5x5": [],
            "raw": None
        }

    if isinstance(resp, dict) and "error" in resp:
        return {
            "dest_present": False,
            "bbox_2d": [0,0,0,0],
            "confidence": 0.0,
            "reason": f"fallback: api error {resp.get('error')}",
            "grid_5x5": [],
            "raw": resp
        }

    raw = (_extract_content(resp) or "").strip()
    parsed = _parse_json_loose(raw) or {}

    def _box_ok(b):
        try:
            x1, y1, x2, y2 = [float(v) for v in b]
            return x2 > x1 and y2 > y1
        except Exception:
            return False

    dest_present = _norm_bool(parsed.get("dest_present", False))
    bbox = parsed.get("bbox_2d") or [0,0,0,0]
    try:
        confidence = float(parsed.get("confidence", 0.0))
        confidence = min(max(confidence, 0.0), 1.0)
    except Exception:
        confidence = 0.0
    reason = (parsed.get("reason") or "").strip() or "N/A"

    grid_5x5 = parsed.get("grid_5x5") or []

    if not dest_present or not _box_ok(bbox):
        return {
            "dest_present": False,
            "bbox_2d": [0,0,0,0],
            "confidence": confidence,
            "reason": reason,
            "grid_5x5": grid_5x5,
            "raw": raw
        }

    side = float(FIXED_CROP_SIDE)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(x1, side-1))
    y1 = max(0.0, min(y1, side-1))
    x2 = max(0.0, min(x2, side-1))
    y2 = max(0.0, min(y2, side-1))

    return {
        "dest_present": True,
        "bbox_2d": [x1,y1,x2,y2],
        "confidence": confidence,
        "reason": reason,
        "grid_5x5": grid_5x5,
        "raw": raw
    }

def _draw_bbox_on_patch(patch, bbox, color=(0,0,255), thickness=3, label=None):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(patch, (x1,y1), (x2,y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt = str(label)
        (tw, th), _ = cv2.getTextSize(txt, font, 0.5, 1)
        cv2.rectangle(patch, (x1, max(0, y1- (th+6))), (x1+tw+10, y1), color, -1)
        cv2.putText(patch, txt, (x1+5, max(0, y1-6)), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

def search_and_reach_destination(
    instr_id, ob, dest_desc, via_desc,
    start_pos_latlng, heading_deg, scale_factor,
    out_dir,
    step_meters=120.0,
    max_steps=3,
    api_key=DEFAULT_API_KEY, url=QWEN_URL, model=QWEN_MODEL
):
    def _haversine_m(lat1, lng1, lat2, lng2):
        R = 6371000.0
        phi1 = math.radians(lat1);  phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lng2 - lng1)
        a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb/2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
        return R * c

    start_pos = [float(start_pos_latlng[0]), float(start_pos_latlng[1])]
    pos = start_pos[:]
    heading = float(heading_deg or 0.0)

    steps_log = []
    found = False
    dest_latlng = None

    predicted_corners_seq = []
    zoom_idx = None
    patch_seq = []

    first_step_pos = None
    fallback_scale = 2.0

    last_grid_5x5 = None

    for k in range(max_steps):
        step_n = k + 1

        corners_k = generate_view_corners_with_scale(
            pos, ob, scale_factor=scale_factor, angle_deg=heading
        )
        predicted_corners_seq.append(np.array(corners_k, dtype=float))

        step_path = os.path.join(
            out_dir,
            f"{instr_id}_step{step_n:02d}_search_view_h{heading:.1f}.jpg"
        )
        patch_k = create_view_image(
            corners_k, ob,
            save_path=step_path,
            out_px=FIXED_CROP_SIDE
        )

        corners_k_s4 = generate_view_corners_with_scale(pos, ob, scale_factor=3, angle_deg=heading)
        s3_path = os.path.join(out_dir, f"{instr_id}_step{step_n:02d}_view_s3_hr.jpg")
        patch_k_s4 = create_view_image(corners_k_s4, ob, save_path=s3_path, out_px=FIXED_CROP_SIDE)
        if patch_k_s4 is not None:
            draw_pos_on_patch(patch_k_s4, corners_k_s4, pos)
            cv2.imwrite(s3_path, patch_k_s4, [cv2.IMWRITE_JPEG_QUALITY, 95])

        corners_k_s8 = generate_view_corners_with_scale(pos, ob, scale_factor=7, angle_deg=heading)
        s7_path = os.path.join(out_dir, f"{instr_id}_step{step_n:02d}_view_s7_hr.jpg")
        patch_k_s8 = create_view_image(corners_k_s8, ob, save_path=s7_path, out_px=FIXED_CROP_SIDE)
        if patch_k_s8 is not None:
            draw_pos_on_patch(patch_k_s8, corners_k_s8, pos)
            cv2.imwrite(s7_path, patch_k_s8, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if patch_k is None:
            steps_log.append({"k": k, "action": "render_fail", "pos": pos})
            pos = _move_forward_geo(pos, heading, meters=step_meters)
            if first_step_pos is None:
                first_step_pos = pos[:]
            continue

        draw_pos_on_patch(patch_k, corners_k, pos)
        cv2.imwrite(step_path, patch_k, [cv2.IMWRITE_JPEG_QUALITY, 95])

        patch_seq.append(patch_k)
        mini_path = os.path.join(
            out_dir,
            f"{instr_id}_step{step_n:02d}_minimap.jpg"
        )
        minimap_bgr = make_minimap_crop_basemap(
            ob,
            corners_seq=predicted_corners_seq,
            img_seq=patch_seq,
            out_path=mini_path,
            pad_m=40.0
        )

        q = qwen_locate_bbox_in_view(
            dest_desc,
            patch_main_bgr=patch_k,
            patch_s4_bgr=patch_k_s4,
            patch_s8_bgr=patch_k_s8,
            minimap_bgr=minimap_bgr,
            api_key=api_key, url=url, model=model,
            restrict_top_half=True,
            prev_grid_5x5=last_grid_5x5
        )

        last_grid_5x5 = q.get("grid_5x5") or last_grid_5x5

        h, w = patch_k.shape[:2]
        steps_log.append({"k": k, "qwen": q, "pos": pos, "w": w, "h": h})

        if q.get("dest_present"):
            bbox = [float(v) for v in (q.get("bbox_2d") or [])][:4]
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                pos = _move_forward_geo(pos, heading, meters=step_meters)
                if first_step_pos is None:
                    first_step_pos = pos[:]
                if step_n >= max_steps and not found:
                    fallback_pos = first_step_pos or _move_forward_geo(start_pos, heading, meters=step_meters)
                    pos = fallback_pos[:]
                    corners_back = generate_view_corners_with_scale(
                        pos, ob,
                        scale_factor=fallback_scale,
                        angle_deg=heading
                    )
                    back_path = os.path.join(
                        out_dir,
                        f"{instr_id}_step{step_n:02d}_return_to_first_step_hr.jpg"
                    )
                    patch_back = create_view_image(
                        corners_back, ob,
                        save_path=back_path,
                        out_px=FIXED_CROP_SIDE
                    )
                    if patch_back is not None:
                        draw_pos_on_patch(patch_back, corners_back, pos)
                        cv2.imwrite(back_path, patch_back, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    predicted_corners_seq.append(np.array(corners_back, dtype=float))
                    steps_log.append({"k": k, "action": "return_to_first_step", "pos": pos})
                    break
                continue

            try:
                side = float(FIXED_CROP_SIDE)
                x1f, y1f, x2f, y2f = bbox
                cx_px = 0.5 * (x1f + x2f)
                cy_px = 0.5 * (y1f + y2f)
                in_lower_half = (cy_px >= side * 0.5)

                if in_lower_half:
                    img_center_ll = _patch_xy_to_latlng(
                        side * 0.5, side * 0.5,
                        corners_k, ob
                    )
                    box_center_ll = _patch_xy_to_latlng(
                        cx_px, cy_px,
                        corners_k, ob
                    )
                    dist_m = _haversine_m(
                        img_center_ll[0], img_center_ll[1],
                        box_center_ll[0], box_center_ll[1]
                    )

                    if dist_m > 120.0:
                        start_box_ll = np.array(ob['gt_path_corners'][0], dtype=float)
                        predicted_corners_seq.append(start_box_ll)
                        steps_log.append({
                            "k": k,
                            "action": "override_to_start_box_on_first_detect_lower_half_far",
                            "dist_m": float(dist_m),
                            "bbox": [x1f, y1f, x2f, y2f]
                        })
                        pos = _corners_center(start_box_ll)
                        found = True
                        zoom_idx = None
                        break
            except Exception as _e:
                steps_log.append({"k": k, "action": "first_detect_rule_error", "error": str(_e)})

            det_vis_path = os.path.join(
                out_dir,
                f"{instr_id}_step{step_n:02d}_det.jpg"
            )
            patch_det = patch_k.copy()
            _draw_bbox_on_patch(
                patch_det, bbox,
                color=(0,0,255), thickness=3,
                label=f"{float(q.get('confidence',0.0)):.2f}"
            )
            cv2.imwrite(det_vis_path, patch_det, [cv2.IMWRITE_JPEG_QUALITY, 95])

            x1f, y1f, x2f, y2f = bbox
            cx_px = 0.5 * (x1f + x2f); cy_px = 0.5 * (y1f + y2f)
            dest_latlng = _patch_xy_to_latlng(cx_px, cy_px, corners_k, ob)
            pos = dest_latlng[:]

            corners_confirm = generate_view_corners_with_scale(
                pos, ob,
                scale_factor=scale_factor,
                angle_deg=heading
            )
            predicted_corners_seq.append(np.array(corners_confirm, dtype=float))

            confirm_path = os.path.join(
                out_dir,
                f"{instr_id}_step{step_n:02d}_dest_confirm_hr.jpg"
            )
            patch_confirm = create_view_image(
                corners_confirm, ob,
                save_path=confirm_path,
                out_px=FIXED_CROP_SIDE
            )
            if patch_confirm is not None:
                draw_pos_on_patch(patch_confirm, corners_confirm, pos)
                cv2.imwrite(confirm_path, patch_confirm, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if patch_confirm is not None:
                patch_seq.append(patch_confirm)

            mini_confirm_path = os.path.join(
                out_dir,
                f"{instr_id}_step{step_n:02d}_minimap_confirm.jpg"
            )
            minimap_confirm_bgr = make_minimap_crop_basemap(
                ob,
                corners_seq=predicted_corners_seq,
                img_seq=patch_seq,
                out_path=mini_confirm_path,
                pad_m=40.0
            )

            corners_confirm_s4 = generate_view_corners_with_scale(pos, ob, scale_factor=3, angle_deg=heading)
            confirm_s3_path = os.path.join(out_dir, f"{instr_id}_step{step_n:02d}_confirm_view_s3_hr.jpg")
            patch_confirm_s4 = create_view_image(corners_confirm_s4, ob, save_path=confirm_s3_path, out_px=FIXED_CROP_SIDE)
            if patch_confirm_s4 is not None:
                draw_pos_on_patch(patch_confirm_s4, corners_confirm_s4, pos)
                cv2.imwrite(confirm_s3_path, patch_confirm_s4, [cv2.IMWRITE_JPEG_QUALITY, 95])

            corners_confirm_s8 = generate_view_corners_with_scale(pos, ob, scale_factor=7, angle_deg=heading)
            confirm_s7_path = os.path.join(out_dir, f"{instr_id}_step{step_n:02d}_confirm_view_s7_hr.jpg")
            patch_confirm_s8 = create_view_image(corners_confirm_s8, ob, save_path=confirm_s7_path, out_px=FIXED_CROP_SIDE)
            if patch_confirm_s8 is not None:
                draw_pos_on_patch(patch_confirm_s8, corners_confirm_s8, pos)
                cv2.imwrite(confirm_s7_path, patch_confirm_s8, [cv2.IMWRITE_JPEG_QUALITY, 95])

            q_confirm = qwen_locate_bbox_in_view(
                dest_desc,
                patch_main_bgr=patch_confirm,
                patch_s4_bgr=patch_confirm_s4,
                patch_s8_bgr=patch_confirm_s8,
                minimap_bgr=minimap_confirm_bgr,
                api_key=api_key, url=url, model=model,
                restrict_top_half=False,
                prev_grid_5x5=last_grid_5x5
            )
            steps_log.append({"k": k, "confirm": q_confirm, "pos": pos})

            last_grid_5x5 = q_confirm.get("grid_5x5") or last_grid_5x5

            confirm_box  = None
            conf_confirm = 0.0

            if q_confirm.get("dest_present"):
                cb = [float(v) for v in q_confirm.get("bbox_2d", [])][:4]
                if len(cb) == 4 and cb[2] > cb[0] and cb[3] > cb[1]:
                    side2 = float(FIXED_CROP_SIDE); eps = 1.0
                    bx1, by1, bx2, by2 = cb
                    bx1 = max(eps, min(bx1, side2-1.0-eps))
                    by1 = max(eps, min(by1, side2-1.0-eps))
                    bx2 = max(eps, min(bx2, side2-1.0-eps))
                    by2 = max(eps, min(by2, side2-1.0-eps))
                    confirm_box  = [bx1, by1, bx2, by2]
                    conf_confirm = float(q_confirm.get("confidence", 0.0))

                    confirm_vis_path = os.path.join(
                        out_dir,
                        f"{instr_id}_step{step_n:02d}_dest_confirm_det.jpg"
                    )
                    patch_confirm_det = patch_confirm.copy()
                    _draw_bbox_on_patch(
                        patch_confirm_det,
                        confirm_box,
                        color=(0,255,0),
                        thickness=3,
                        label=f"C:{conf_confirm:.2f}"
                    )
                    cv2.imwrite(
                        confirm_vis_path,
                        patch_confirm_det,
                        [cv2.IMWRITE_JPEG_QUALITY, 95]
                    )

            x1f, y1f, x2f, y2f = [float(v) for v in bbox]
            if confirm_box is not None:
                bx1, by1, bx2, by2 = confirm_box
                use_box = [bx1, by1, bx2, by2]
                cx_use = 0.5 * (bx1 + bx2)
                cy_use = 0.5 * (by1 + by2)
                center_latlng = _patch_xy_to_latlng(
                    cx_use, cy_use,
                    corners_confirm, ob
                )
            else:
                use_box = [x1f, y1f, x2f, y2f]
                center_latlng = dest_latlng

            def _minimal_square_uv_from_bbox(bbox, side=FIXED_CROP_SIDE, pad_px=0):
                x1, y1, x2, y2 = [float(v) for v in bbox]
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                edge = max(w, h) * 0.5
                edge += float(pad_px)

                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)

                def _clamp_square(cx, cy, half, S):
                    left  = cx - half; right = cx + half
                    top   = cy - half; bot   = cy + half

                    if left < 0:      cx += -left
                    if right > S-1:   cx -= (right - (S-1))
                    if top  < 0:      cy += -top
                    if bot  > S-1:    cy -= (bot - (S-1))

                    half = min(half, cx, S-1-cx, cy, S-1-cy)

                    left  = cx - half; right = cx + half
                    top   = cy - half; bot   = cy + half
                    return cx, cy, half, (left, top, right, bot)

                cx, cy, edge, (l, t, r, b) = _clamp_square(
                    cx, cy, edge, float(side)
                )
                return [(l, t), (r, t), (r, b), (l, b)]

            corners_base = corners_confirm if (confirm_box is not None) else corners_k

            ZOOM_PAD_PX = 0
            sq_uv = _minimal_square_uv_from_bbox(
                use_box,
                side=FIXED_CROP_SIDE,
                pad_px=ZOOM_PAD_PX
            )

            corners_zoom = [
                _patch_xy_to_latlng(u, v, corners_base, ob)
                for (u, v) in sq_uv
            ]

            zoom_path = os.path.join(
                out_dir,
                f"{instr_id}_step{step_n:02d}_dest_zoom_hr.jpg"
            )
            patch_zoom = create_view_image(
                corners_zoom, ob,
                save_path=zoom_path,
                out_px=FIXED_CROP_SIDE
            )
            predicted_corners_seq.append(np.array(corners_zoom, dtype=float))
            zoom_idx = len(predicted_corners_seq) - 1

            found = True
            break

        else:
            pos = _move_forward_geo(pos, heading, meters=step_meters)
            if first_step_pos is None:
                first_step_pos = pos[:]

            if step_n >= max_steps and not found:
                fallback_pos = first_step_pos or _move_forward_geo(start_pos, heading, meters=step_meters)
                pos = fallback_pos[:]
                corners_back = generate_view_corners_with_scale(
                    pos, ob,
                    scale_factor=fallback_scale,
                    angle_deg=heading
                )
                back_path = os.path.join(
                    out_dir,
                    f"{instr_id}_step{step_n:02d}_return_to_first_step_hr.jpg"
                )
                patch_back = create_view_image(
                    corners_back, ob,
                    save_path=back_path,
                    out_px=FIXED_CROP_SIDE
                )
                if patch_back is not None:
                    draw_pos_on_patch(patch_back, corners_back, pos)
                    cv2.imwrite(back_path, patch_back, [cv2.IMWRITE_JPEG_QUALITY, 95])
                predicted_corners_seq.append(np.array(corners_back, dtype=float))
                steps_log.append({"k": k, "action": "return_to_first_step", "pos": pos})
                break

    search_json_path = os.path.join(out_dir, f"{instr_id}_search.json")
    try:
        out = {
            "instr_id": instr_id,
            "found": bool(found),
            "dest_latlng": dest_latlng,
            "final_pos": pos,
            "heading_deg": heading,
            "step_meters": step_meters,
            "max_steps": max_steps,
            "steps": steps_log
        }
        with open(search_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] failed to write search JSON: {search_json_path} -> {e}")

    return found, dest_latlng, predicted_corners_seq, zoom_idx

def hook_after_turn_and_crop(row, ob, instr_id, pos, new_heading, scale_factor, out_dir,
                             api_key=DEFAULT_API_KEY, url=QWEN_URL, model=QWEN_MODEL):
    dest_desc = (row.get("dest") or "").strip()
    via_desc  = (row.get("via") or "").strip()
    if via_desc.lower() in ("none","null","na","n/a",""):
        via_desc = None
    print(f"[DEST_DESC] {instr_id}: {dest_desc}")

    found, dest_latlng, predicted_corners_seq, zoom_idx = search_and_reach_destination(
        instr_id=instr_id,
        ob=ob,
        dest_desc=dest_desc,
        via_desc=via_desc,
        start_pos_latlng=pos,
        heading_deg=new_heading,
        scale_factor=scale_factor,
        out_dir=out_dir,
        step_meters=120,
        max_steps=3,
        api_key=api_key,
        url=url,
        model=model
    )

    if found:
        print(f"[OK][DEST] {instr_id}: destination found, center approx {dest_latlng}")
    else:
        print(f"[INFO][DEST] {instr_id}: destination not found within max steps; search log written.")

    return predicted_corners_seq, zoom_idx

def run_turn_and_crop(anno_dir=ANNO_DIR, dataset_dir=DATASET_DIR, split=SPLIT,
                      results_csv=RESULTS_CSV, out_dir=OUT_DIR,
                      scale_factor=SCALE_FACTOR):
    try:
        pass
    except Exception as e:
        raise RuntimeError("Cannot import DataLoader or ANDHNavBatch; check dependencies and PYTHONPATH.") from e

    os.makedirs(out_dir, exist_ok=True)

    tif_dataset_dir = os.path.join(dataset_dir, 'train_images')
    env = ANDHNavBatch(
        anno_dir=anno_dir,
        dataset_dir=tif_dataset_dir,
        splits=[split],
        tokenizer=None,
        max_instr_len=512,
        batch_size=1,
        seed=0,
        full_traj=False
    )
    loader = DataLoader(env, batch_size=1)
    preds = {}
    _metrics_acc = {"final_iou": [], "path_length": []}

    id2row = {}
    with open(results_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            instr_id = (r.get("instr_id") or "").strip()
            if instr_id:
                id2row[instr_id] = r

    processed = 0
    skipped = 0

    for _batch_idx, _ in enumerate(loader):
        obs_list = env._get_obs(t=0)
        if not obs_list:
            continue
        ob = obs_list[0]
        ob["dataset_dir"] = dataset_dir

        map_name  = ob.get("map_name", "")
        route_idx = ob.get("route_index", "")
        instr_id  = f"{map_name}__{route_idx}"

        if instr_id not in id2row:
            skipped += 1
            continue

        row = id2row[instr_id]
        instruction = row.get("instruction", "")
        move_dir = (row.get("move_dir") or "").strip()

        start_corners = np.array(ob['gt_path_corners'][0])
        pos     = np.mean(start_corners, axis=0)
        heading = float(ob.get("starting_angle", 0.0) or 0.0)

        corners0 = generate_view_corners_with_scale(pos, ob, scale_factor=scale_factor, angle_deg=heading)
        step0_hr = os.path.join(out_dir, f"{instr_id}_h{heading:.1f}_step00_start_hr.jpg")
        patch0   = create_view_image(corners0, ob, save_path=step0_hr, out_px=FIXED_CROP_SIDE)
        if patch0 is None:
            print(f"[WARN] start patch failed, skipping: {instr_id}")
            skipped += 1
            continue
        draw_pos_on_patch(patch0, corners0, pos)
        cv2.imwrite(step0_hr, patch0, [cv2.IMWRITE_JPEG_QUALITY, 95])

        corners_abs  = generate_view_corners_with_scale(pos, ob, scale_factor=scale_factor, angle_deg=None)
        step0_abs_hr = os.path.join(out_dir, f"{instr_id}_step00_abs_north_hr.jpg")
        patch_abs    = create_view_image(corners_abs, ob, save_path=step0_abs_hr, out_px=FIXED_CROP_SIDE)

        if move_dir.lower() == "land":
            use_img = patch_abs if patch_abs is not None else patch0
            clock_str, reason, meta = decide_turn_from_land(instruction, use_img, view="absolute", heading_deg=heading)
            if isinstance(meta, dict) and meta.get("rel_deg") is not None:
                clock_str = angle_to_clock(meta["rel_deg"])
        else:
            clock_str = move_dir
            reason = ""
            meta = {}

        delta_deg  = clock_to_angle_deg(clock_str)
        new_heading = (heading + delta_deg) % 360.0

        corners1 = generate_view_corners_with_scale(pos, ob, scale_factor=scale_factor, angle_deg=new_heading)
        step1_hr = os.path.join(out_dir, f"{instr_id}_turn_{clock_str.replace(':','-')}_h{new_heading:.1f}_step01_after_turn_hr.jpg")
        patch1   = create_view_image(corners1, ob, save_path=step1_hr, out_px=FIXED_CROP_SIDE)
        if patch1 is not None:
            draw_pos_on_patch(patch1, corners1, pos)
            cv2.imwrite(step1_hr, patch1, [cv2.IMWRITE_JPEG_QUALITY, 95])

        dest_desc = (row.get("dest") or "").strip()
        print(f"[DEST_DESC] {instr_id}: {dest_desc}")
        forward_raw = str(row.get("forward", "")).strip().lower()
        forward_bool = forward_raw in ("true", "1", "yes", "y", "t")

        predicted_boxes = []
        if isinstance(corners0, (np.ndarray, list)):
            predicted_boxes.append(np.array(corners0, dtype=float))
        if isinstance(corners1, (np.ndarray, list)):
            predicted_boxes.append(np.array(corners1, dtype=float))

        zoom_highlight_idx = None

        if dest_desc.lower() == "destination":

            start_box = np.array(ob['gt_path_corners'][0], dtype=float)
            start_ctr = np.array(_corners_center(start_box), dtype=float)

            if forward_bool:
                end_ctr = np.array(_move_forward_geo(start_ctr.tolist(), new_heading, meters=60.0), dtype=float)
                delta = end_ctr - start_ctr
                final_box = (start_box + delta).astype(float)
            else:
                final_box = start_box.astype(float)

            predicted_boxes.append(final_box)

            out_traj_img = os.path.join(out_dir, f"{instr_id}_traj_boxes.jpg")
            save_traj_boxes_debug_image(
                tif_dataset_dir=tif_dataset_dir,
                map_name=map_name,
                ob=ob,
                predicted_boxes=predicted_boxes,
                out_path=out_traj_img,
                zoom_highlight_idx=None
            )

        else:
            try:
                predicted_seq_from_search, zoom_idx_local = hook_after_turn_and_crop(
                    row=row,
                    ob=ob,
                    instr_id=instr_id,
                    pos=pos,
                    new_heading=new_heading,
                    scale_factor=scale_factor,
                    out_dir=out_dir,
                    api_key=DEFAULT_API_KEY,
                    url=QWEN_URL,
                    model=QWEN_MODEL
                )
            except Exception as e:
                print(f"[WARN] search error: {instr_id} -> {e}")
                predicted_seq_from_search, zoom_idx_local = [], None

            for c in (predicted_seq_from_search or []):
                predicted_boxes.append(np.array(c, dtype=float))
            if zoom_idx_local is not None:
                zoom_highlight_idx = 2 + zoom_idx_local

            out_traj_img = os.path.join(out_dir, f"{instr_id}_traj_boxes.jpg")
            save_traj_boxes_debug_image(
                tif_dataset_dir=tif_dataset_dir,
                map_name=map_name,
                ob=ob,
                predicted_boxes=predicted_boxes,
                out_path=out_traj_img,
                zoom_highlight_idx=zoom_highlight_idx
            )

        goal_corners = np.array(ob['gt_path_corners'][-1], dtype=float)
        progress     = [float(compute_iou(pb, goal_corners)) for pb in predicted_boxes] if predicted_boxes else []
        trajectory   = [_corners_center(pb) for pb in predicted_boxes]
        path_corners = [(pb.tolist(), float(ob.get("starting_angle", 0.0) or 0.0)) for pb in predicted_boxes]

        final_iou   = float(progress[-1]) if progress else 0.0
        path_length = len(predicted_boxes)
        success     = final_iou > 0.3

        traj_entry = {
            "instr_id": instr_id,
            "trajectory": trajectory,
            "path_corners": path_corners,
            "progress": progress,
            "gt_progress": progress[:],
            "gt_path_corners": ob["gt_path_corners"],
            "reasoning": [],
            "final_iou": final_iou,
            "path_length": path_length,
            "success": success,
        }

        preds[instr_id] = traj_entry
        _metrics_acc["final_iou"].append(final_iou)
        _metrics_acc["path_length"].append(path_length)

        rel_info = ""
        abs_info = ""
        if move_dir.lower() == "land" and isinstance(meta, dict):
            if meta.get("rel_deg") is not None:
                rel_info = f" | rel≈{float(meta['rel_deg']):.1f}°"
                q_deg = clock_to_angle_deg(clock_str)
                diff = abs(((q_deg - float(meta['rel_deg']) + 540) % 360) - 180)
                if diff > 7.6:
                    print(f"[WARN] quantization mismatch: rel={float(meta['rel_deg']):.1f}°, clock={clock_str}({q_deg:.1f}°)")
            if meta.get("abs_deg") is not None:
                abs_info = f" | abs={float(meta['abs_deg']):.1f}°"

        if patch1 is None:
            print(f"[WARN] turn patch render failed: {instr_id} ({clock_str}) | reason: {reason[:120]}{rel_info}{abs_info}")
        else:
            print(f"[OK] {instr_id}: heading {heading:.1f} -> {new_heading:.1f} "
                  f"by {clock_str} ({delta_deg:.1f}°){rel_info}{abs_info} | reason: {reason[:120]}")

        analysis_path = os.path.join(out_dir, f"{instr_id}_analysis.json")
        try:
            out_json = {
                "instr_id": instr_id,
                "map_name": map_name,
                "route_idx": route_idx,
                "instruction": instruction,
                "heading_deg": heading,
                "clock": clock_str,
                "delta_deg": delta_deg,
                "new_heading_deg": new_heading,
                "used_view": "absolute" if move_dir.lower() == "land" else "heading",
                "reason": reason
            }
            if meta:
                out_json["meta"] = meta
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(out_json, f, ensure_ascii=False, indent=2)
        except Exception as _e:
            print(f"[WARN] failed to write analysis JSON: {analysis_path} -> {_e}")

        processed += 1

    print(f"\nDone: generated {processed}  entries; skipped {skipped}  entries. Output dir: {out_dir}")

    if _metrics_acc["final_iou"]:
        success_rate    = sum(1 for v in _metrics_acc["final_iou"] if v > 0.3) / len(_metrics_acc["final_iou"])
        avg_iou         = float(np.mean(_metrics_acc["final_iou"]))
        avg_path_length = float(np.mean(_metrics_acc["path_length"]))
    else:
        success_rate = 0.0
        avg_iou = 0.0
        avg_path_length = 0.0

    metrics = {
        "success_rate": success_rate,
        "avg_iou": avg_iou,
        "avg_path_length": avg_path_length,
    }

    result_file = os.path.join(out_dir, "turn_and_crop_eval_results.json")
    def _to_jsonable(x):
        if hasattr(x, "tolist"): return x.tolist()
        if isinstance(x, dict):  return {k: _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, list):  return [_to_jsonable(v) for v in x]
        return x

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable({"predictions": preds, "metrics": metrics}), f, ensure_ascii=False, indent=2)

    try:
        score_summary, detailed_metrics = env.eval_metrics(preds, human_att_eval=False)
        print("\n" + "="*50)
        print("Natural Path Eval summary:", score_summary)
        txt_file = os.path.join(out_dir, "turn_and_crop_score_summary.txt")
        with open(txt_file, "w", encoding="utf-8") as f:
            if isinstance(score_summary, dict):
                for k, v in score_summary.items():
                    f.write(f"{k}: {v}\n")
            else:
                f.write(str(score_summary))
        print(f"Score summary saved to: {txt_file}")
    except Exception as e:
        detailed_metrics = {}
        print(f"[WARN] env.eval_metrics failed: {e}")

    print(f"Results saved to: {result_file}")

    return preds, detailed_metrics

def create_view_image(view_corners, ob, save_path=None, out_px=768):
    try:
        map_path = os.path.join(ob['dataset_dir'], 'train_images', f"{ob['map_name']}.tif")
        if not os.path.exists(map_path):
            print(f"[WARN] map file not found: {map_path}")
            return None

        sat_map = cv2.imread(map_path, cv2.IMREAD_COLOR)
        if sat_map is None:
            print(f"[WARN] failed to read map: {map_path}")
            return None

        h, w = sat_map.shape[:2]
        lat_min, lng_min = ob['gps_botm_left']
        lat_max, lng_max = ob['gps_top_right']

        src = []
        for lat, lng in view_corners:
            x = (lng - lng_min) / (lng_max - lng_min) * w
            y = (lat_max - lat) / (lat_max - lat_min) * h
            src.append([x, y])
        src = np.array(src, dtype=np.float32)
        src[:, 0] = np.clip(src[:, 0], 0, w - 1)
        src[:, 1] = np.clip(src[:, 1], 0, h - 1)

        if cv2.contourArea(src.astype(np.float32)) < 1.0:
            print("[WARN] view area too small")
            return None

        out_side = int(768 if (out_px is None or out_px == 'auto') else out_px)

        dst = np.array([[0,0],[out_side-1,0],[out_side-1,out_side-1],[0,out_side-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)

        patch_hr = cv2.warpPerspective(sat_map, M, (out_side, out_side), flags=cv2.INTER_LANCZOS4)
        if patch_hr is None or patch_hr.size == 0:
            print("[WARN] perspective warp returned empty")
            return None

        if save_path:
            cv2.imwrite(save_path, patch_hr, [cv2.IMWRITE_JPEG_QUALITY, 98])

        return patch_hr
    except Exception as e:
        print(f"[ERROR] crop failed: {e}")
        return None

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    run_turn_and_crop()