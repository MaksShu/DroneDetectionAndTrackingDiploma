import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cm = np.array([[5671, 1005],
               [671,   0]])
labels = ["drone", "background"]

annot = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = '-' if cm[i, j] == 0 else str(cm[i, j])

plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    cm,
    annot=annot,            
    fmt="",                 
    cmap="Blues",
    annot_kws={"fontsize": 24, "fontweight": "bold"},
    cbar=False
)

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 
ax.set_xlabel("True", fontsize=20, labelpad=10)
ax.set_ylabel("Predicted", fontsize=20)

ax.set_xticks(np.arange(len(labels)) + 0.5)
ax.set_xticklabels(labels, fontsize=16)
ax.set_yticks(np.arange(len(labels)) + 0.5)
ax.set_yticklabels(labels, fontsize=16)

plt.tight_layout()
plt.show()