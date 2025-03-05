import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from LoadData import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Standardize them
X_scaled = StandardScaler().fit_transform(X)  # Standardization

# Perform PCA
pca = PCA(n_components=3)  # We need at least 3 components
Z = pca.fit_transform(X_scaled)  # Projected data

# Get cylinder values for coloring
cylinders = data['cylinders']
unique_cylinders = np.unique(cylinders)
colors = sns.color_palette("Set1", len(unique_cylinders))

# Define projections
projections = [(0, 1), (1, 2), (0, 2)]  # (PC1 vs PC2), (PC2 vs PC3), (PC1 vs PC3)
titles = ["PC1 vs PC2", "PC2 vs PC3", "PC1 vs PC3"]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

for ax, (i, j), title in zip(axes, projections, titles):
    ax.set_title(title)
    for cyl, color in zip(unique_cylinders, colors):
        mask = cylinders == cyl  # Select data points with this cylinder count
        ax.scatter(Z[mask, i], Z[mask, j], color=color, label=f"Cylinders: {cyl}", alpha=0.6, edgecolors="k", s=40)
    ax.set_xlabel(f"PC{i+1}")
    ax.set_ylabel(f"PC{j+1}")
    ax.legend()

# Display all plots
plt.tight_layout()
plt.show()


# Assign colors explicitly
colors = np.full(df.shape[0], "", dtype=object)  # Empty array to store colors
colors[low_mpg_mask] = "red"      # Low MPG (<18)
colors[medium_mpg_mask] = "orange"  # Medium MPG (18-30)
colors[high_mpg_mask] = "green"     # High MPG (≥30)


# MPG category labels for legend
mpg_categories = {
    "red": "Low MPG (<18)",
    "orange": "Medium MPG (18-30)",
    "green": "High MPG (≥30)"
}

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

for ax, (i, j), title in zip(axes, projections, titles):
    ax.set_title(title)
    
    # Scatter plot using explicitly assigned colors
    ax.scatter(Z[:, i], Z[:, j], c=colors, alpha=0.6, edgecolors="k", s=40)
    
    ax.set_xlabel(f"PC{i+1}")
    ax.set_ylabel(f"PC{j+1}")

# Create custom legend
from matplotlib.patches import Patch
legend_patches = [Patch(color=color, label=label) for color, label in mpg_categories.items()]
axes[0].legend(handles=legend_patches, title="MPG Category")

# Display all plots
plt.tight_layout()
plt.show()
