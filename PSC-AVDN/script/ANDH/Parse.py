import os
import sys
import re
import csv
import json
import base64
import math
from pathlib import Path
from typing import Iterable, Tuple, List, Optional

import numpy as np
import cv2
import requests
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))

from env import ANDHNavBatch

CFGPU_URL   = ""
CFGPU_MODEL = ""
CFGPU_API_TOKEN = os.getenv("API_TOKEN", "")   
ANNO_DIR = str(PROJECT_ROOT / "datasets" / "AVDN" / "annotations")
DATASET_DIR = str(PROJECT_ROOT / "datasets" / "AVDN")
SPLIT       = "val_seen"
PRED_DIR = str(PROJECT_ROOT / "preds_out")
SCALE_FACTOR = 3.0
def generate_view_corners_with_scale(center_point, ob, scale_factor=1.0, angle_deg=None):
    center_point = np.array(center_point, dtype=float).reshape(2,)
    lat_min, lng_min = ob['gps_botm_left']
    lat_max, lng_max = ob['gps_top_right']
    h, w = ob['map_size'][:2]

    lat_per_px = (lat_max - lat_min) / h
    lng_per_px = (lng_max - lng_min) / w

    base_pixels = 224 * float(scale_factor)
    half_lat = (base_pixels / 2) * lat_per_px
    half_lng = (base_pixels / 2) * lng_per_px

    if angle_deg is None:
        return np.array([
            [center_point[0] + half_lat, center_point[1] - half_lng],
            [center_point[0] + half_lat, center_point[1] + half_lng],
            [center_point[0] - half_lat, center_point[1] + half_lng],
            [center_point[0] - half_lat, center_point[1] - half_lng],
        ], dtype=float)

    theta = np.radians(float(angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    local = np.array([
        [+half_lat, -half_lng],
        [+half_lat, +half_lng],
        [-half_lat, +half_lng],
        [-half_lat, -half_lng],
    ], dtype=float)
    R = np.array([[cos_t, -sin_t],[sin_t, cos_t]], dtype=float)
    rot = local @ R.T
    return rot + center_point

def create_view_image(view_corners, ob, save_path=None):
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

        dst = np.array([[0,0],[223,0],[223,223],[0,223]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        patch = cv2.warpPerspective(sat_map, M, (224,224))
        if patch is None or patch.size == 0:
            print("[WARN] perspective warp returned empty")
            return None

        if save_path:
            cv2.imwrite(save_path, patch)
        return patch
    except Exception as e:
        print(f"[ERROR] crop failed: {e}")
        return None

def clock_to_angle_deg(clock_str):
    s = str(clock_str).strip().replace("点","").replace("方向","")
    m = re.fullmatch(r'(\d{1,2})(?::(\d{1,2}))?$', s)
    if not m:
        alias = {"forward":"12","front":"12","ahead":"12","straight":"12",
                 "right":"3","left":"9","back":"6","backward":"6",
                 "front-right":"1:30","front-left":"10:30",
                 "back-right":"4:30","back-left":"7:30"}
        s = alias.get(s.lower(), "12")
        m = re.fullmatch(r'(\d{1,2})(?::(\d{1,2}))?$', s)
    hour = int(m.group(1)) % 12
    minute = int(m.group(2) or 0)
    return (hour * 30.0 + minute * 0.5) % 360.0

def analyze_instruction_with_prompt(instruction: str,
                                    model: str = CFGPU_MODEL,
                                    api_token: str = CFGPU_API_TOKEN,
                                    base_url: str = CFGPU_URL) -> str:
    system_prompt = (
        "You are an expert in drone navigation command analysis. Please structurally understand and break down the received instructions, and output standardized results."
        "Only output two conclusions, do not explain or output the thought process:"
        "Next movement direction (if based on landmarks, output Land here): (clock direction or Land)  Destination description:"
        "[General Principles]"
        "1) Only use the [INS] (instruction) in the user message as the basis. Initial orientation is assumed to be 12:00."
        "2) Direction source discrimination: If based on the drone (e.g., your 3 o'clock / ahead / behind / left / turn left then go forward), output clock direction;"
        "If based on landmarks (e.g., the northeastern part of the landfill / south side of the stadium / west of the bridge), output Land."
        "If neither based on landmarks nor the drone, but only standalone absolute directions (e.g., north/east/south/west/northeast/southwest/southeast/northwest, etc.), output the corresponding degrees (N=0°, NE=45°, E=90°, SE=135°, S=180°, SW=225°, W=270°, NW=315°)"
        "3) Next movement direction = the 'synthesized heading' after completing necessary turns, just before starting to move, not breaking turns into multiple steps, nor the static direction of the destination relative to the drone."
        "4) Error tolerance: Recognize non-standard spellings (oclock/o' clock/o clok, forword, lef, etc.), colloquial and grammatical defects."
        "5) Minimum clock granularity: 15° (supports 1:15, 3:30, 4:45, etc.)."
        "6) Output a brief destination description (landmark + specific part/building, etc.)."
        "7) When a sentence contains both 'relative to the drone' and 'relative to landmarks', first determine if it is landmark-based (e.g., 'the <direction> of the <landmark>'), if yes output Land; only when definitely based on the drone as reference, output clock direction."
        "8) Sequential synthesis rule (core): Find the most recent verb that causes 'movement' (head/go/move/proceed/fly/continue);"
        "Synthesize the previous turning/adjustment verbs (turn/rotate/pivot/backwards/clockwise/counterclockwise, etc.) in sequence into a final heading,"
        "This final heading is the 'next movement direction'."
        "9) Conversational reference: If [INS] is a follow-up sentence in a dialogue, incorporate contextual phrases (e.g., the last building there)."
        "10) Clock mapping reference: N=12:00; NE=1:30; E=3:00; SE=4:30; S=6:00; SW=7:30; W=9:00; NW=10:30;"
        "slight left/right≈±15°; sharp left/right≈±90°; back/behind/turn backwards=+180°; other slight turn instructions can also output +/-15°"
        "11) If unable to determine a clear movement heading but can confirm it is landmark-based, output Land for direction; if neither landmarks nor a determinable heading, conservatively output 12:00."
        "12) right in front of you / right ahead / straight ahead / just ahead / in front of you / ahead → 12:00 (here right is for emphasis, not indicating right side)."
        "13) Destination description should be in full English and may include waypoint information as reference; if the destination is described based on waypoints or reference objects, need a complete description for subsequent visual positioning."
        "14) If there is no specific destination description, or it is judged that the current location is the destination, directly output 'destination' for the 'Destination description' field."
        "[Strictly follow the output format, no other formats allowed: fixed, only one line, strictly prohibit outputting any other text]"
        "Next movement direction: <direction>  Destination description: <brief description or destination>"
    )

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "stream": False,
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ],
    }

    try:
        resp = requests.post(base_url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Request to cfgpu failed: {e}") from e

    data = resp.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        if isinstance(data, dict):
            for key in ("content", "text", "output"):
                if key in data and isinstance(data[key], str):
                    return data[key]
        raise RuntimeError(f"Unexpected response schema: {data}")

def run_min_rot(anno_dir, dataset_dir, split=SPLIT, pred_dir=PRED_DIR,
                scale_factor=SCALE_FACTOR):
    pattern_en = re.compile(
        r"Next\s*movement\s*direction\s*:\s*(?P<dir>.*?)\s+"
        r"Destination\s*description\s*:\s*(?P<dest>.*?)(?:\s+"
        r"(?:Via|Waypoints|Through)\s*:\s*(?P<via>.*))?\s*$",
        re.IGNORECASE | re.S
    )
    pattern_zh = re.compile(
        r"下一步移动方向[^：]*：\s*(?P<dir>.*?)\s+"
        r"(?:目的地描述|目的地)[^：]*：\s*(?P<dest>.*?)(?:\s+"
        r"(?:途径点|经由)[^：]*：\s*(?P<via>.*))?\s*$",
        re.S
    )

    def parse_model_output(output: str):
        output = (output or "").strip()
        m = pattern_en.search(output) or pattern_zh.search(output)
        if not m:
            return "", output, ""
        move_dir = (m.group("dir") or "").strip()
        dest     = (m.group("dest") or "").strip()
        via      = (m.group("via") or "").strip()
        return move_dir, dest, via

    os.makedirs(pred_dir, exist_ok=True)
    step_dir = os.path.join(pred_dir, "stepwise_views")
    os.makedirs(step_dir, exist_ok=True)
    results_csv = os.path.join(pred_dir, "parsing_results_raw.csv")

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

    try:
        total_samples = len(loader)
    except Exception:
        total_samples = "unknown"
    print(f"ANDHNavBatch loaded with {total_samples} instructions, using splits: {split}")

    rows = []

    for batch_idx, _ in enumerate(tqdm(loader, desc="samples")):
        obs_list = env._get_obs(t=0)
        assert len(obs_list) == 1
        ob = obs_list[0]
        ob["dataset_dir"] = dataset_dir

        map_name = ob.get("map_name", "")
        route_idx = ob.get("route_index", "")
        instr_id = f"{map_name}__{route_idx}"
        instruction = ob.get("instructions", "")

        print(f"[SAMPLE {batch_idx+1}] map_name={map_name} route={route_idx}")

        start_corners = np.array(ob['gt_path_corners'][0])
        pos = np.mean(start_corners, axis=0)

        heading = float(ob.get("starting_angle", 0.0) or 0.0)

        corners0 = generate_view_corners_with_scale(
            pos, ob, scale_factor=scale_factor, angle_deg=heading
        )
        patch0 = create_view_image(corners0, ob)

        if patch0 is None:
            print("[WARN] start patch crop failed, skipping sample.")
            continue

        lat_min, lng_min = ob['gps_botm_left']
        lat_max, lng_max = ob['gps_top_right']
        h, w = ob['map_size'][:2]

        x_map = (pos[1] - lng_min) / (lng_max - lng_min) * w
        y_map = (lat_max - pos[0]) / (lat_max - lat_min) * h
        src_point = np.array([[x_map, y_map]], dtype=np.float32)

        src = []
        for lat, lng in corners0:
            x = (lng - lng_min) / (lng_max - lng_min) * w
            y = (lat_max - lat) / (lat_max - lat_min) * h
            src.append([x, y])
        src = np.array(src, dtype=np.float32)
        dst = np.array([[0, 0], [223, 0], [223, 223], [0, 223]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)

        dst_point = cv2.perspectiveTransform(src_point[None, :, :], M)[0, 0]
        px, py = int(dst_point[0]), int(dst_point[1])

        if 0 <= px < 224 and 0 <= py < 224:
            cv2.circle(patch0, (px, py), radius=5, color=(0, 255, 0), thickness=-1)

        debug_name = f"{instr_id}_step00_start_with_point.jpg"
        cv2.imwrite(os.path.join(step_dir, debug_name), patch0)

        try:
            output = analyze_instruction_with_prompt(
                instruction,
                model=CFGPU_MODEL,
                api_token=CFGPU_API_TOKEN,
                base_url=CFGPU_URL
            )

        except Exception as e:
            print(f"[ERROR] analyze_instruction_with_prompt failed: {e}")
            output = ""

        print("Model raw output:\n", output)

        move_dir, dest, via = parse_model_output(output)

        rows.append({
            "instr_id": instr_id,
            "map_name": map_name,
            "route_idx": route_idx,
            "instruction": instruction,
            "move_dir": move_dir,
            "dest": dest,
            "via": via,
            "angle": heading,
            "raw_output": (output or "").strip()
        })

    with open(results_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "instr_id", "map_name", "route_idx", "instruction",
            "move_dir", "dest", "via", "angle", "raw_output"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written to: {results_csv}")
    print(f"Screenshot dir: {step_dir}")

def safe_read_csv(path: Path, encodings: Iterable[str] = ("utf-8-sig", "utf-8", "gbk", "cp1252", "latin1")) -> Tuple[pd.DataFrame, str]:
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, enc
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to read CSV with provided encodings.")

CLOCK_MAP = {
    0: "12:00",
    45: "1:30",
    90: "3:00",
    135: "4:30",
    180: "6:00",
    225: "7:30",
    270: "9:00",
    315: "10:30",
}

DEGREE_MARK = re.compile(r"^(\d+)\s*°$")

def degree_to_clock(angle: float) -> str:
    a = float(angle) % 360.0
    nearest = (int(((a + 22.5) // 45) * 45)) % 360
    return CLOCK_MAP.get(nearest, CLOCK_MAP[0])

def convert_move_dir_row(move_dir_value, angle_value) -> str:
    move_dir = str(move_dir_value).strip()
    try:
        angle = float(angle_value)
    except Exception:
        angle = 0.0

    m = DEGREE_MARK.match(move_dir)
    if m:
        abs_angle = int(m.group(1)) % 360
        rel_angle = (abs_angle - angle) % 360
        return degree_to_clock(rel_angle)
    else:
        return move_dir

INS_BLOCK = re.compile(r"\[INS\](.*?)(?:\[/INS\]|$)", flags=re.IGNORECASE | re.DOTALL)
FORWARD_PATTERN = re.compile(
    r"(?:\b(?:go|move|fly|proceed|continue)\s*(?:straight|forward|forword)\b)"
    r"|(?:\b(?:straight(?:\s*ahead)?|forward|forword)\b)",
    flags=re.IGNORECASE
)

def extract_ins(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = INS_BLOCK.search(text)
    return (m.group(1) if m else text).strip()

def has_forward(ins_text: str) -> bool:
    return bool(FORWARD_PATTERN.search(ins_text or ""))

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
def clean_text(x):
    if not isinstance(x, str):
        return x
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = CONTROL_CHARS.sub("", x)
    x = re.sub(r"[ \t]+", " ", x)
    return x.strip()

def postprocess_csv(pred_dir=PRED_DIR):
    input_csv = Path(pred_dir) / "parsing_results_raw.csv"
    intermediate_out = Path(pred_dir) / "parsing_results_clock.csv"
    final_out = Path(pred_dir) / "parsing_results.csv"

    df, used_enc = safe_read_csv(input_csv)
    if "move_dir" in df.columns and "angle" in df.columns:
        df["move_dir"] = [convert_move_dir_row(md, ang) for md, ang in zip(df["move_dir"], df["angle"])]
    else:
        missing = [c for c in ("move_dir", "angle") if c not in df.columns]
        print(f"Warning: missing columns for move_dir conversion: {missing}")
    df.to_csv(intermediate_out, index=False, encoding="utf-8-sig")
    print("Intermediate saved to:", intermediate_out)

    df2, used_enc2 = safe_read_csv(intermediate_out)
    if "dest" in df2.columns:
        dest = df2["dest"].astype(str).str.strip().str.lower()
    else:
        dest = pd.Series([], dtype=str)
    is_dest = dest.eq("destination")
    ins = df2.get("instruction", pd.Series([], dtype=str)).astype(str).map(extract_ins)
    forward_bool = is_dest & ins.map(has_forward)
    df2["forward_bool"] = forward_bool
    df2["forward"] = forward_bool.map(lambda v: "TRUE" if bool(v) else "FALSE")
    for col in df2.columns:
        if df2[col].dtype == object:
            df2[col] = df2[col].astype(str).map(clean_text)
    preferred_order = [
        "instr_id", "map_name", "route_idx", "instruction",
        "move_dir", "dest", "via", "forward", "forward_bool",
        "angle", "raw_output"
    ]
    cols = [c for c in preferred_order if c in df2.columns] + [c for c in df2.columns if c not in preferred_order]
    df2 = df2[cols]
    df2.to_csv(
        final_out,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\r\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    print("Read encoding (step1):", used_enc)
    print("Read encoding (step2):", used_enc2)
    print("Final saved to:", final_out)

def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    run_min_rot(ANNO_DIR, DATASET_DIR, SPLIT, PRED_DIR, scale_factor=SCALE_FACTOR)
    postprocess_csv(PRED_DIR)

if __name__ == "__main__":
    main()
