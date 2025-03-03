import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from LoadData import *

# Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained Variance Ratio
explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Scree Plot
plt.figure(figsize=(7, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.axhline(y=0.95, color='r', linestyle='--', label="95% Variance Explained")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Scree Plot")
plt.legend()
plt.grid()
plt.show()

# Find the number of components for 95% variance
num_components = np.argmax(explained_variance >= 0.95) + 1
print(f"Number of Principal Components needed for 95% variance: {num_components}")