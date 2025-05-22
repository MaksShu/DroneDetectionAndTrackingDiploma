import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("..\Faster RCNN\\training_log.csv")

sns.set_style("whitegrid")

plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["map"], label="mAP", marker='o')
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["early_stop_counter"], label="Early Stop Counter", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Counter")
plt.legend()
plt.show()