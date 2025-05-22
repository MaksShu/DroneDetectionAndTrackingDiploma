import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
BASE_DIR = Path('.')
SPLITS = ["train", "val"]
NEW_ROOT = BASE_DIR / "filtered"
KEEP_CLASS = 2

# prepare target dirs
for sub in ("images", "labels"):
    for split in SPLITS:
        (NEW_ROOT / sub / split).mkdir(parents=True, exist_ok=True)

# extensions to try when matching an image to its .txt
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

for split in SPLITS:
    src_imgs = BASE_DIR / "images" / split
    src_lbls = BASE_DIR / "labels" / split
    dst_imgs = NEW_ROOT   / "images" / split
    dst_lbls = NEW_ROOT   / "labels" / split

    for lbl_path in tqdm(src_lbls.glob("*.txt")):
        lines = [l for l in lbl_path.read_text().splitlines() if l.strip()]
        classes = [int(l.split()[0]) for l in lines]
        
        if any(c != KEEP_CLASS for c in classes):
            # Find and copy the image
            stem = lbl_path.stem
            img_src = None
            for ext in IMG_EXTS:
                cand = src_imgs / f"{stem}{ext}"
                if cand.exists():
                    img_src = cand
                    break
            if img_src is None:
                print(f"⚠️  no image found for {stem}, skipping")
                continue
            shutil.copy2(img_src, dst_imgs / img_src.name)

            # Keep only KEEP_CLASS lines and remap to class 0
            new_lines = ["0 " + " ".join(l.split()[1:]) for l in lines if int(l.split()[0]) == KEEP_CLASS]

            # Write output label
            out_lbl = dst_lbls / f"{stem}.txt"
            out_lbl.write_text("\n".join(new_lines))

print("Done. Filtered dataset is in ./filtered/")