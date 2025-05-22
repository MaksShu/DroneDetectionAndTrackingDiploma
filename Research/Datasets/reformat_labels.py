from pathlib import Path

# Configuration
BASE_DIR = Path('.')
LABEL_ROOT = BASE_DIR / "labels"
TARGET_CLASS = "0"

for txt_path in LABEL_ROOT.rglob("*.txt"):
    lines = txt_path.read_text().splitlines()
    # Keep only lines with target class
    new_lines = [line for line in lines
                 if line.strip() and line.split()[0] == TARGET_CLASS]

    # Overwrite file (empty if no matching lines)
    txt_path.write_text("\n".join(new_lines))

print("Done – all labels not class 0 have been removed.")