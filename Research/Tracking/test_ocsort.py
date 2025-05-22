import os
import time
import zipfile
import torch

import cv2
import numpy as np
import gdown
from tqdm import tqdm
from ocsort.ocsort import OCSort
import motmetrics as mm


def download_mot17():
    if not os.path.exists('MOT17/train'):
        print("Downloading MOT17...")
        gdown.download("https://motchallenge.net/data/MOT17.zip", "MOT17.zip", quiet=False)
        with zipfile.ZipFile("MOT17.zip", 'r') as z:
            z.extractall(".")
        print("MOT17 downloaded and extracted.")


def load_sequences():
    seqs = []
    for seq in os.listdir('MOT17/train'):
        if 'FRCNN' not in seq: continue
        img_dir = os.path.join('MOT17/train', seq, 'img1')
        det_file = os.path.join('MOT17/train', seq, 'det', 'det.txt')
        gt_file = os.path.join('MOT17/train', seq, 'gt',  'gt.txt')
        if os.path.isdir(img_dir) and os.path.exists(det_file) and os.path.exists(gt_file):
            seqs.append({
                'name': seq,
                'img_dir': img_dir,
                'det_file': det_file,
                'gt_data': np.loadtxt(gt_file, delimiter=',')
            })
    return seqs


def load_detections(det_path):
    data = np.loadtxt(det_path, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

def evaluate_ocsort_on_dets(img_dir, dets, gt_data, tracker):
    acc = mm.MOTAccumulator(auto_id=True)
    fps_list = []

    frames = np.unique(dets[:, 0].astype(int))
    for frame_id in frames:
        img = cv2.imread(os.path.join(img_dir, f"{frame_id:06d}.jpg"))
        if img is None:
            continue

        sel = dets[dets[:, 0] == frame_id]

        bbox_scores = []
        for r in sel:
            l, t, w, h, s = r[2], r[3], r[4], r[5], r[6]
            bbox_scores.append([l, t, l + w, t + h, s, 0])
        det_array = np.array(bbox_scores, dtype=float)

        det_tensor = torch.from_numpy(det_array).float()

        t0 = time.time()
        online_targets = tracker.update(det_tensor, None)
        dt = time.time() - t0
        fps_list.append(1.0 / dt if dt > 0 else 0)

        bbs, ids = [], []
        for trk in online_targets:
            x1, y1, x2, y2, tid = trk[:5]
            bbs.append([x1, y1, x2, y2])
            ids.append(int(tid))
        bbs = np.array(bbs, float)
        ids = np.array(ids, int)

        mask = gt_data[:, 0] == frame_id
        gt_ids = gt_data[mask, 1].astype(int)
        gt_boxes = gt_data[mask, 2:6].astype(float).copy()
        gt_boxes[:, 2] += gt_boxes[:, 0]
        gt_boxes[:, 3] += gt_boxes[:, 1]

        if gt_boxes.size and bbs.size:
            dist = mm.distances.iou_matrix(gt_boxes, bbs, max_iou=0.5)
        else:
            dist = np.zeros((gt_boxes.shape[0], bbs.shape[0]), float)

        acc.update(
            gt_ids.tolist() if gt_ids.ndim > 0 else ([int(gt_ids)] if gt_ids.size else []),
            ids.tolist() if ids.ndim > 0 else ([int(ids)] if ids.size else []),
            dist
        )

    m = mm.metrics.create().compute(acc, metrics=['mota', 'idf1'], name='ocsort')
    return m.loc['ocsort'], np.mean(fps_list)


if __name__ == '__main__':
    download_mot17()
    sequences = load_sequences()

    ocsort_tracker = OCSort()

    all_mota, all_idf1, all_fps = [], [], []
    for seq in tqdm(sequences, desc="Sequence", unit="seq"):
        dets = load_detections(seq['det_file'])
        metrics, fps = evaluate_ocsort_on_dets(seq['img_dir'], dets, seq['gt_data'], ocsort_tracker)
        all_mota.append(metrics['mota'])
        all_idf1.append(metrics['idf1'])
        all_fps.append(fps)

    print("\n=== OC-SORT on MOT17 (precomputed dets) ===")
    print(f"Average MOTA = {np.mean(all_mota):.3f}")
    print(f"Average IDF1 = {np.mean(all_idf1):.3f}")
    print(f"Average FPS  = {np.mean(all_fps):.2f}")