"""
WIDER Face Easy/Medium/Hard AP evaluation (pure NumPy).

Adapted from wondervictor / Pytorch_Retinaface evaluation protocol.
Predictions use WIDER format: x, y, w, h, score.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


def bbox_overlaps(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """IoU overlaps with +1 inclusive WIDER/VOC box convention. Vectorized."""
    boxes = boxes.astype(np.float64, copy=False)
    query_boxes = query_boxes.astype(np.float64, copy=False)
    box_areas = (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1.0) * (
        query_boxes[:, 3] - query_boxes[:, 1] + 1.0
    )
    iw = (
        np.minimum(boxes[:, None, 2], query_boxes[None, :, 2])
        - np.maximum(boxes[:, None, 0], query_boxes[None, :, 0])
        + 1.0
    )
    ih = (
        np.minimum(boxes[:, None, 3], query_boxes[None, :, 3])
        - np.maximum(boxes[:, None, 1], query_boxes[None, :, 1])
        + 1.0
    )
    inter = np.maximum(iw, 0.0) * np.maximum(ih, 0.0)
    ua = box_areas[:, None] + query_areas[None, :] - inter
    return inter / np.maximum(ua, 1e-12)


def get_gt_boxes(gt_dir: str):
    gt_mat = loadmat(os.path.join(gt_dir, "wider_face_val.mat"))
    hard_mat = loadmat(os.path.join(gt_dir, "wider_hard_val.mat"))
    medium_mat = loadmat(os.path.join(gt_dir, "wider_medium_val.mat"))
    easy_mat = loadmat(os.path.join(gt_dir, "wider_easy_val.mat"))
    return (
        gt_mat["face_bbx_list"],
        gt_mat["event_list"],
        gt_mat["file_list"],
        hard_mat["gt_list"],
        medium_mat["gt_list"],
        easy_mat["gt_list"],
    )


def read_pred_file(filepath: str) -> Tuple[str, np.ndarray]:
    with open(filepath, "r") as f:
        lines = f.readlines()
    img_file = lines[0].rstrip("\n\r")
    boxes = []
    for line in lines[2:]:
        parts = line.rstrip("\r\n").split(" ")
        if not parts or parts[0] == "":
            continue
        boxes.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return img_file.split("/")[-1], np.array(boxes, dtype=np.float64) if boxes else np.zeros((0, 5))


def get_preds(pred_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    boxes: Dict[str, Dict[str, np.ndarray]] = {}
    events = sorted(os.listdir(pred_dir))
    for event in tqdm(events, desc="Reading predictions"):
        event_dir = os.path.join(pred_dir, event)
        if not os.path.isdir(event_dir):
            continue
        current_event = {}
        for imgtxt in os.listdir(event_dir):
            if not imgtxt.endswith(".txt"):
                continue
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip(".jpg")] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred: Dict[str, Dict[str, np.ndarray]]) -> None:
    max_score, min_score = 0.0, 1.0
    for event in pred.values():
        for v in event.values():
            if len(v) == 0:
                continue
            max_score = max(float(np.max(v[:, -1])), max_score)
            min_score = min(float(np.min(v[:, -1])), min_score)
    diff = max_score - min_score
    if diff <= 0:
        return
    for event in pred.values():
        for v in event.values():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score) / diff


def image_eval(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray, iou_thresh: float):
    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)
    for h in range(_pred.shape[0]):
        gt_overlap = overlaps[h]
        max_overlap = gt_overlap.max()
        max_idx = int(gt_overlap.argmax())
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1
        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num: int, pred_info: np.ndarray, proposal_list: np.ndarray, pred_recall: np.ndarray):
    pr_info = np.zeros((thresh_num, 2), dtype=np.float64)
    for t in range(thresh_num):
        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[: r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num: int, pr_curve: np.ndarray, count_face: int):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0] if pr_curve[i, 0] > 0 else 0
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face if count_face > 0 else 0
    return _pr_curve


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def evaluation(pred_dir: str, gt_path: str, iou_thresh: float = 0.5) -> List[float]:
    pred = get_preds(pred_dir)
    norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ["easy", "medium", "hard"]
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps: List[float] = []

    for setting_id in range(3):
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2), dtype=np.float64)
        for i in tqdm(range(event_num), desc=f"Processing {settings[setting_id]}"):
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred.get(event_name, {})
            sub_gt_list = gt_list[i][0]
            gt_bbx_list = facebox_list[i][0]
            for j in range(len(img_list)):
                img_key = str(img_list[j][0][0])
                pred_info = pred_list.get(img_key, np.zeros((0, 5), dtype=np.float64))
                gt_boxes = gt_bbx_list[j][0].astype("float")
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)
                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)
                pr_curve += img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
        aps.append(voc_ap(pr_curve[:, 1], pr_curve[:, 0]))

    print("==================== Results ====================")
    print(f"Easy   Val AP: {aps[0]:.4f}")
    print(f"Medium Val AP: {aps[1]:.4f}")
    print(f"Hard   Val AP: {aps[2]:.4f}")
    print("=================================================")
    return aps
