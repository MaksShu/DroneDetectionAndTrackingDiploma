import shutil
import random
from pathlib import Path

# Configuration
BASE_DIR = Path('.')
NEW_ROOT = BASE_DIR / "test_filtered"
SPLIT = "test"
SAMPLE_PCT = 0.50
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(SEED)

src_imgs = list((BASE_DIR / "images" / SPLIT).iterdir())
src_imgs = [p for p in src_imgs if p.suffix.lower() in IMG_EXTS]

# Sample images
n_sample = int(len(src_imgs) * SAMPLE_PCT)
sampled = set(random.sample(src_imgs, n_sample))

dst_imgs = NEW_ROOT / "images" / SPLIT
dst_lbls = NEW_ROOT / "labels" / SPLIT
dst_imgs.mkdir(parents=True, exist_ok=True)
dst_lbls.mkdir(parents=True, exist_ok=True)

for img_path in sampled:
    # Copy image
    shutil.copy2(img_path, dst_imgs / img_path.name)

    # Process label: keep only class-2 → remap to 0
    stem = img_path.stem
    lbl_src = BASE_DIR / "labels" / SPLIT / f"{stem}.txt"
    out_txt = []

    if lbl_src.exists():
        for line in lbl_src.read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == "2":
                parts[0] = "0"
                out_txt.append(" ".join(parts))

    (dst_lbls / f"{stem}.txt").write_text("\n".join(out_txt))

print(f"Sampled {n_sample} images (50%) into {dst_imgs} and remapped labels into {dst_lbls}")