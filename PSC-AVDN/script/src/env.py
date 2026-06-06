import json
import os
import numpy as np
import random
from collections import defaultdict
import cv2
import torch
import shapely
import shapely.geometry
from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely.ops import nearest_points


def compute_iou(a, b):
    a = np.array(a)
    poly1 = Polygon(a).convex_hull
    b = np.array(b)
    poly2 = Polygon(b).convex_hull
    union_poly = np.concatenate((a, b))
    if not poly1.intersects(poly2):
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                iou = 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print("shapely.geos.TopologicalError occured, iou set to 0")
            iou = 0
    return iou


def get_direction(start, end):
    vec = np.array(end) - np.array(start)
    _angle = 0
    if vec[1] > 0:
        _angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90
    elif vec[1] < 0:
        _angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90 + 180
    else:
        if np.sign(vec[0]) == 1:
            _angle = 90
        else:
            _angle = 270
    _angle = (360 - _angle + 90) % 360
    return _angle


def name_the_direction(_angle):
    if _angle > 337.5 or _angle < 22.5:
        return "north"
    elif np.abs(_angle - 45) <= 22.5:
        return "northeast"
    elif np.abs(_angle - 135) <= 22.5:
        return "southeast"
    elif np.abs(_angle - 90) <= 22.5:
        return "east"
    elif np.abs(_angle - 180) <= 22.5:
        return "south"
    elif np.abs(_angle - 315) <= 22.5:
        return "northwest"
    elif np.abs(_angle - 225) <= 22.5:
        return "southwest"
    elif np.abs(_angle - 270) <= 22.5:
        return "west"


class ANDHNavBatch(torch.utils.data.IterableDataset):
    def __init__(
        self,
        anno_dir,
        dataset_dir,
        splits,
        tokenizer=None,
        max_instr_len=512,
        batch_size=64,
        seed=0,
        full_traj=False,
    ):
        self.dataset_dir = dataset_dir
        self.data = []
        for split in splits:
            new_data = json.load(open(os.path.join(anno_dir, "%s_data.json" % split)))
            if full_traj == False:
                for item in new_data:
                    item["angle"] = round(item["angle"]) % 360
                    for i in range(len(item["gt_path_corners"])):
                        item["gt_path_corners"][i] = np.array(
                            item["gt_path_corners"][i]
                        )
                    item["instructions"] = item["instructions"].lower()
                    item["pre_dialogs"] = " ".join(item["pre_dialogs"]).lower()
                    self.data.append(item)
            print(
                "%s loaded with %d instructions, using splits: %s"
                % (self.__class__.__name__, len(new_data), split)
            )
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        self.ix = 0
        self.batch_size = batch_size
        self.map_batch = {}
        self.attention_map_batch = {}

    def size(self):
        return len(self.data)

    def gps_to_img_coords(self, gps, ob):
        gps_botm_left = ob["gps_botm_left"]
        gps_top_right = ob["gps_top_right"]
        lng_ratio = ob["lng_ratio"]
        lat_ratio = ob["lat_ratio"]
        return (
            int(round((gps[1] - gps_botm_left[1]) / lat_ratio)),
            int(round((gps_top_right[0] - gps[0]) / lat_ratio)),
        )

    def next_batch(self):
        batch_size = self.batch_size
        for ix in range(0, len(self.data), batch_size):
            batch = self.data[ix : ix + batch_size]
            if len(batch) < batch_size:
                ix = batch_size - len(batch)
                batch += self.data[:ix]
            self.batch = batch
            used_map_names = []
            for i in range(batch_size):
                used_map_names.append(self.batch[i]["map_name"])
                if not used_map_names[-1] in self.map_batch.keys():
                    im = cv2.imread(
                        os.path.join(self.dataset_dir, f"{used_map_names[-1]}.tif"), 1
                    )
                    lng_ratio = self.batch[i]["lng_ratio"]
                    lat_ratio = self.batch[i]["lat_ratio"]
                    im_resized = cv2.resize(
                        im,
                        (int(im.shape[1] * lng_ratio / lat_ratio), im.shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                    self.map_batch[used_map_names[-1]] = im_resized
                    attention_map = np.zeros(
                        (im_resized.shape[0], im_resized.shape[1], 3), np.uint8
                    )
                    self.attention_map_batch[used_map_names[-1]] = attention_map
            to_be_deleted = []
            for k in self.map_batch:
                if not k in used_map_names:
                    to_be_deleted.append(k)
            for k in to_be_deleted:
                del self.map_batch[k]
                del self.attention_map_batch[k]
            max_instruction_length = 0
            for i in range(batch_size):
                if len(self.batch[i]["instructions"]) > max_instruction_length:
                    max_instruction_length = len(self.batch[i]["instructions"])
            self.max_instruction_length = max_instruction_length
            yield used_map_names

    def __iter__(self):
        return self.next_batch()

    def _get_obs(self, corners=None, directions=None, t=None, shortest_teacher=False):
        obs = []
        for i in range(self.batch_size):
            item = self.batch[i]
            if t == None:
                t_input = 0
            else:
                if t < len(item["gt_path_corners"]):
                    t_input = t
                else:
                    t_input = len(item["gt_path_corners"]) - 1
            if corners is None:
                view_area_corners = item["gt_path_corners"][t_input]
            else:
                view_area_corners = corners[i]
            width = 224
            height = 224
            dst_pts = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype="float32",
            )
            view_area_corners = np.array(view_area_corners)
            img_coord_view_area_corners = view_area_corners
            for xx in range(view_area_corners.shape[0]):
                img_coord_view_area_corners[xx] = self.gps_to_img_coords(
                    view_area_corners[xx], item
                )
            img_coord_view_area_corners = np.array(
                img_coord_view_area_corners, dtype="float32"
            )
            M = cv2.getPerspectiveTransform(img_coord_view_area_corners, dst_pts)
            im_view = cv2.warpPerspective(
                self.map_batch[item["map_name"]], M, (width, height)
            )
            gt_saliency = cv2.warpPerspective(
                self.attention_map_batch[item["map_name"]], M, (width, height)
            )
            gt_saliency = (
                np.asarray(cv2.cvtColor(gt_saliency, cv2.COLOR_BGR2GRAY)) / 255
            )
            obs_item = {
                "map_name": item["map_name"],
                "map_size": self.map_batch[item["map_name"]].shape,
                "route_index": item["route_index"],
                "gps_botm_left": item["gps_botm_left"],
                "gps_top_right": item["gps_top_right"],
                "lng_ratio": item["lng_ratio"],
                "lat_ratio": item["lat_ratio"],
                "starting_angle": item["angle"],
                "current_view": im_view,
                "gt_saliency": gt_saliency,
                "attention_list": item.get("attention_list", []),
                "gt_path_corners": item["gt_path_corners"],
                "view_area_corners": view_area_corners,
                "instructions": item["instructions"],
                "pre_dialogs": item["pre_dialogs"],
            }
            dest_quad = item.get("destination", None)
            if (
                isinstance(dest_quad, (list, tuple))
                and len(dest_quad) == 4
                and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in dest_quad)
            ):
                dest_quad = [[float(p[0]), float(p[1])] for p in dest_quad]
            else:
                dest_quad = None
            obs_item["destination"] = dest_quad
            obs.append(obs_item)
        return obs

    def _eval_item(self, gt_path, gt_corners, path, corners, progress):
        scores = {}
        scores["trajectory_lengths"] = np.sum(
            [np.linalg.norm(a - b) for a, b in zip(path[:-1], path[1:])]
        )
        scores["trajectory_lengths"] = scores["trajectory_lengths"] * 11.13 * 1e4
        gt_whole_lengths = (
            np.sum([np.linalg.norm(a - b) for a, b in zip(gt_path[:-1], gt_path[1:])])
            * 11.13
            * 1e4
        )
        gt_net_lengths = np.linalg.norm(gt_path[0] - gt_path[-1]) * 11.13 * 1e4
        scores["iou"] = progress[-1]
        scores["gp"] = (
            gt_net_lengths - np.linalg.norm(path[-1] - gt_path[-1]) * 11.13 * 1e4
        )
        scores["oracle_gp"] = (
            gt_net_lengths
            - np.min([np.linalg.norm(path[x] - gt_path[-1]) for x in range(len(path))])
            * 11.13
            * 1e4
        )
        scores["success"] = float(progress[-1] >= 0.4)
        _center = np.mean(gt_corners[-1], axis=0)
        _point = Point(_center)
        _poly = Polygon(np.array(corners[-1]))
        if not _poly.contains(_point):
            scores["success"] = float(0)
        _center = np.mean(corners[-1], axis=0)
        _point = Point(_center)
        _poly = Polygon(np.array(gt_corners[-1]))
        if not _poly.contains(_point):
            scores["success"] = float(0)
        scores["oracle_success"] = float(any(np.array(progress) > 0.4))
        scores["gt_length"] = gt_whole_lengths
        scores["spl"] = (
            scores["success"]
            * gt_net_lengths
            / max(scores["trajectory_lengths"], gt_net_lengths, 0.01)
        )
        return scores

    def eval_metrics(self, preds, human_att_eval=False):
        metrics = defaultdict(list)
        if human_att_eval == True:
            for k in preds.keys():
                if "human_att_performance" in preds[k].keys():
                    metrics["human_att_performance"] += preds[k][
                        "human_att_performance"
                    ]
                    nss = np.mean(preds[k]["nss"])
                    if nss == nss:
                        metrics["nss"].append(nss)
            metrics["human_att_performance"] = np.mean(
                metrics["human_att_performance"], axis=0
            )
            metrics["nss"] = np.mean(metrics["nss"])
            if metrics["nss"] == metrics["nss"]:
                avg_metrics = {
                    "HA_precision": metrics["human_att_performance"][0],
                    "HA_recall": metrics["human_att_performance"][0],
                    "nss": metrics["nss"],
                }
            else:
                avg_metrics = {"HA_precision": 0, "HA_recall": 0, "nss": 0}
            return avg_metrics, metrics
        for k in preds.keys():
            item = preds[k]
            instr_id = item["instr_id"]
            dia_number = 0
            if "num_dia" in preds[k].keys():
                dia_number = preds[k]["num_dia"]
            traj = [np.mean(x[0], axis=0) for x in item["path_corners"]]
            corners = [np.array(x[0]) for x in item["path_corners"]]
            progress = [x for x in item["gt_progress"]]
            gt_corners = [np.array(x) for x in item["gt_path_corners"]]
            gt_trajs = [np.mean(x, axis=0) for x in item["gt_path_corners"]]
            traj_scores = self._eval_item(gt_trajs, gt_corners, traj, corners, progress)
            for k, v in traj_scores.items():
                if k == "iou" and traj_scores["success"]:
                    metrics[k].append(v)
                else:
                    metrics[k].append(v)
            if dia_number == 1:
                metrics["success_1"].append(traj_scores["success"])
                metrics["spl_1"].append(traj_scores["spl"])
                metrics["gp_1"].append(traj_scores["gp"])
            elif dia_number == 2:
                metrics["success_2"].append(traj_scores["success"])
                metrics["spl_2"].append(traj_scores["spl"])
                metrics["gp_2"].append(traj_scores["gp"])
            else:
                metrics["success_else"].append(traj_scores["success"])
                metrics["spl_else"].append(traj_scores["spl"])
                metrics["gp_else"].append(traj_scores["gp"])
            if traj_scores["trajectory_lengths"] > 150:
                metrics["success_long"].append(traj_scores["success"])
                metrics["spl_long"].append(traj_scores["spl"])
                metrics["gp_long"].append(traj_scores["gp"])
            else:
                metrics["success_short"].append(traj_scores["success"])
                metrics["spl_short"].append(traj_scores["spl"])
                metrics["gp_short"].append(traj_scores["gp"])
            metrics["instr_id"].append(instr_id)
        avg_metrics = {
            "lengths": np.mean(metrics["trajectory_lengths"]),
            "sr": np.mean(metrics["success"]) * 100,
            "oracle_sr": np.mean(metrics["oracle_success"]) * 100,
            "spl": np.mean(metrics["spl"]) * 100,
            "gp": np.mean(metrics["gp"]),
            "oracle_gp": np.mean(metrics["oracle_gp"]),
            "gt_length": np.mean(metrics["gt_length"]),
            "iou": np.mean(metrics["iou"]),
        }
        if len(metrics["success_1"]) != 0:
            avg_metrics["num_1"] = len(metrics["success_1"])
            avg_metrics["spl_1"] = np.mean(metrics["spl_1"]) * 100
            avg_metrics["sr_1"] = np.mean(metrics["success_1"]) * 100
            avg_metrics["gp_1"] = np.mean(metrics["gp_1"])
        if len(metrics["success_2"]) != 0:
            avg_metrics["num_2"] = len(metrics["success_2"])
            avg_metrics["spl_2"] = np.mean(metrics["spl_2"]) * 100
            avg_metrics["sr_2"] = np.mean(metrics["success_2"]) * 100
            avg_metrics["gp_2"] = np.mean(metrics["gp_2"])
        if len(metrics["success_else"]) != 0:
            avg_metrics["num_else"] = len(metrics["success_else"])
            avg_metrics["spl_else"] = np.mean(metrics["spl_else"]) * 100
            avg_metrics["sr_else"] = np.mean(metrics["success_else"]) * 100
            avg_metrics["gp_else"] = np.mean(metrics["gp_else"])
        return avg_metrics, metrics
