# Projecting onto the principal components
from LoadData import *
import matplotlib.pyplot as plt 
from scipy.linalg import svd

# Subtract mean value from data
Y = X - np.ones((N, 1)) * X.mean(0)

# PCA by computing SVD of Y
U, S, Vh = svd(Y, full_matrices=False)
V = Vh.T  # Transpose to get correct V

# Project the centered data onto principal component space
Z = Y @ V

# Define projections
projections = [(0, 1), (1, 2), (0, 2)]  # (PC1 vs PC2), (PC2 vs PC3), (PC1 vs PC3)
titles = ["PC1 vs PC2", "PC2 vs PC3", "PC1 vs PC3"]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns



for ax, (i, j), title in zip(axes, projections, titles):
    ax.set_title(title)
    for c in range(C):
        class_mask = y == c  # Select indices belonging to class c
        ax.plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.5, label=classNames[c])
    ax.set_xlabel(f"PC{i+1}")
    ax.set_ylabel(f"PC{j+1}")
    ax.legend()

# Display all plots
plt.tight_layout()
plt.show()
