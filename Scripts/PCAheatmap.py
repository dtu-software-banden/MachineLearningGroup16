import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from LoadData import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Standardize the data (important for PCA if features have different scales)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=3)  # Choose how many components you want
X_pca = pca.fit_transform(X_scaled)

# Get the principal component loadings (eigenvectors)
loadings = pca.components_

# Convert to DataFrame for easier visualization
pc_df = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(loadings.shape[0])], index=attributeNames)

# Plot heatmap of component loadings
plt.figure(figsize=(8, 5))
sns.heatmap(pc_df, annot=True, cmap="coolwarm", center=0)
plt.title("Principal Component Loadings")
plt.xlabel("Principal Components")
plt.ylabel("Original Features")
plt.show()
