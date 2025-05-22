import os
import glob
import time
import zipfile

import cv2
import numpy as np
import gdown
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort
import motmetrics as mm


def download_mot17():
    if not os.path.exists('MOT17/train'):
        print("Downloading MOT17...")
        gdown.download("https://motchallenge.net/data/MOT17.zip", "MOT17.zip", quiet=False)
        with zipfile.ZipFile("MOT17.zip", 'r') as z:
            z.extractall(".")
        print("MOT17 downloaded and extracted.")


download_mot17()

deep_sort = DeepSort(
    max_age=30,
)

if hasattr(deep_sort, 'encoder') and hasattr(deep_sort.encoder, 'model'):
    import torch

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    deep_sort.encoder.model.to(device)

sequences = []
for seq in os.listdir('MOT17/train'):
    if 'FRCNN' not in seq:
        continue
    seq_dir = os.path.join('MOT17/train', seq)
    img_dir = os.path.join(seq_dir, 'img1')
    det_file = os.path.join(seq_dir, 'det', 'det.txt')
    gt_file = os.path.join(seq_dir, 'gt', 'gt.txt')
    if os.path.isdir(img_dir) and os.path.exists(det_file) and os.path.exists(gt_file):
        sequences.append({
            'name': seq,
            'img_dir': img_dir,
            'det_file': det_file,
            'gt_data': np.loadtxt(gt_file, delimiter=',')
        })


def load_detections(det_path):
    data = np.loadtxt(det_path, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def evaluate_deepsort_on_dets(img_dir, dets, gt_data):
    acc = mm.MOTAccumulator(auto_id=True)
    fps_list = []

    frames = np.unique(dets[:, 0].astype(int))
    for frame_id in frames:
        img_path = os.path.join(img_dir, f"{frame_id:06d}.jpg")
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        sel = dets[dets[:, 0] == frame_id]

        raw = []
        for r in sel:
            left, top, w, h, conf = r[2], r[3], r[4], r[5], r[6]
            raw.append(([float(left), float(top), float(w), float(h)], float(conf)))

        t0 = time.time()
        tracks = deep_sort.update_tracks(raw, None, frame)
        dt = time.time() - t0
        fps_list.append(1.0 / dt if dt > 0 else 0)

        bbs, ids = [], []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x, y, w, h = t.to_ltwh()
            bbs.append([x, y, x + w, y + h])
            ids.append(int(t.track_id))
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

        gt_ids_list = gt_ids.tolist() if gt_ids.ndim > 0 else ([int(gt_ids)] if gt_ids.size else [])
        ids_list = ids.tolist() if ids.ndim > 0 else ([int(ids)] if ids.size else [])
        acc.update(gt_ids_list, ids_list, dist)

    metrics = mm.metrics.create().compute(acc, metrics=['mota', 'idf1'], name='deepsort')
    return metrics.loc['deepsort'], np.mean(fps_list)


if __name__ == '__main__':
    all_mota, all_idf1, all_fps = [], [], []
    for seq in tqdm(sequences, desc="Sequence", unit="seq"):
        dets = load_detections(seq['det_file'])
        m, fps = evaluate_deepsort_on_dets(seq['img_dir'], dets, seq['gt_data'])
        all_mota.append(m['mota'])
        all_idf1.append(m['idf1'])
        all_fps.append(fps)

    print("\n=== DeepSORT on MOT17 (precomputed dets) ===")
    print(f"Average MOTA = {np.mean(all_mota):.3f}")
    print(f"Average IDF1 = {np.mean(all_idf1):.3f}")
    print(f"Average FPS  = {np.mean(all_fps):.2f}")
