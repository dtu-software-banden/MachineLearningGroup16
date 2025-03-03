import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
filename = "auto-mpg-revised-ml.data.csv"
df = pd.read_csv(filename, delimiter='\t')

# Select relevant attributes
attributes = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration','origin']

# Pairplot with color coding based on 'origin'
sns.pairplot(df[attributes], diag_kind='kde', hue='origin', palette='viridis', plot_kws={'alpha': 0.6, 's': 20})
plt.suptitle("Pairwise Scatter Plots of Attributes (Color by origin)", y=1.02)
plt.show()
