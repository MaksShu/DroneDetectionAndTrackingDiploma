import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(".\\results.csv")

df["epoch"] = df["epoch"].astype(int)
df["train_loss"] = df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]
df["val_loss"] = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]

sns.set_style("whitegrid")


plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
plt.plot(df["epoch"], df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"], label="Validation Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(np.arange(0, df["epoch"].max() + 1, step=2))
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP", marker='o')
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.xticks(np.arange(0, df["epoch"].max() + 1, step=2))
plt.legend()
plt.show()

