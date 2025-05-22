import sys
from pathlib import Path
from PIL import Image
from statistics import mean
from collections import Counter
from fractions import Fraction

DATASET_ROOT = Path("..\\Datasets\\Evaluating Dataset")
IMAGES_ROOT = DATASET_ROOT / "images"
LABELS_ROOT = DATASET_ROOT / "labels"

TOP_N_RES = 8
TOP_N_RATIO = 8

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def gather_images(root: Path):
    """Recursively collect all image files under root."""
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def main():
    img_files = gather_images(IMAGES_ROOT)
    if not img_files:
        print(f"No images found in {IMAGES_ROOT}")
        sys.exit(1)

    res_counts = Counter()
    obj_counts = []
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    aspect_strs = []
    missing_labels = []
    empty_labels = []

    for img_path in img_files:
        rel = img_path.relative_to(IMAGES_ROOT)
        lbl_path = LABELS_ROOT / rel.with_suffix('.txt')

        with Image.open(img_path) as im:
            w, h = im.size
        res_counts[(w, h)] += 1

        if not lbl_path.exists():
            missing_labels.append(lbl_path)
            obj_counts.append(0)
            continue

        lines = [l for l in lbl_path.read_text().splitlines() if l.strip()]
        if not lines:
            empty_labels.append(lbl_path)
            obj_counts.append(0)
            continue

        obj_counts.append(len(lines))
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            _, xc, yc, nw, nh = map(float, parts)
            bw = nw * w
            bh = nh * h
            area_pct = bw * bh / (w * h) * 100

            bbox_widths.append(bw)
            bbox_heights.append(bh)
            bbox_areas.append(area_pct)

            if bh > 0:
                ratio = bw / bh
                frac = Fraction(ratio).limit_denominator(32)
                aspect_strs.append(f"{frac.numerator}/{frac.denominator}")

    total = len(img_files)
    print(f"Images analysed    : {total}\n")

    print(f"Top {TOP_N_RES} resolutions:")
    for (w, h), cnt in res_counts.most_common(TOP_N_RES):
        print(f"  {w}×{h} : {cnt}")

    print(f"\nObjects per image  : min={min(obj_counts)}  max={max(obj_counts)}  mean={mean(obj_counts):.2f}\n")

    if bbox_widths:
        print(f"BBox width  (px)   : min={min(bbox_widths):.0f}  mean={mean(bbox_widths):.1f}  max={max(bbox_widths):.0f}")
        print(f"BBox height (px)   : min={min(bbox_heights):.0f}  mean={mean(bbox_heights):.1f}  max={max(bbox_heights):.0f}")
        print(f"BBox area %        : min={min(bbox_areas):.2f}%  mean={mean(bbox_areas):.2f}%  max={max(bbox_areas):.2f}%")

        ratio_counts = Counter(aspect_strs)
        print(f"\nTop {TOP_N_RATIO} aspect ratios (fractional):")
        for frac_str, cnt in ratio_counts.most_common(TOP_N_RATIO):
            print(f"  {frac_str} : {cnt}")

    if missing_labels:
        print(f"\nMissing label files: {len(missing_labels)}")
    if empty_labels:
        print(f"Empty label files  : {len(empty_labels)}")


if __name__ == '__main__':
    main()
