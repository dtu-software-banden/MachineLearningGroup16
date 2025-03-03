import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
filename = "auto-mpg-revised-ml.data.csv"
df = pd.read_csv(filename, delimiter='\t')

# Select relevant attributes
attributes = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration','origin']

# Pairplot
sns.pairplot(df[attributes], diag_kind='kde', plot_kws={'alpha': 0.5, 's': 15})  # 'kde' for smooth histograms
plt.suptitle("Pairwise Scatter Plots of Attributes", y=1.02)
plt.show()