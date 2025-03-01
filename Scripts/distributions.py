import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
from ucimlrepo import fetch_ucirepo 

# Fetch dataset
auto_mpg = fetch_ucirepo(id=9) 

# Data (as pandas dataframes)
X = auto_mpg.data.features 
y = auto_mpg.data.targets 

# Print attributes and check for missing data
print(auto_mpg.variables)
print(X.isnull().sum())  # 6 missing data

# Drop rows with missing values in 'horsepower'
X_cleaned = X.dropna(subset=['horsepower'])

############################################################################################

# Target attribute:
title = "Model Year"
var = X_cleaned['model_year']


# Calculate mean and standard deviation
mean = var.mean()
median = var.median()

# Plot distribution for variable
sns.histplot(var, kde=True, bins=20, label="KDE (Density Curve)") 


# Add a vertical line for the mean and std
plot.axvline(mean, color='r', linestyle='solid', label=f'Mean: {mean:.2f}')
plot.axvline(median, color='yellow', linestyle='solid', label=f'Median: {median:.2f}')

# Customize plot
plot.title('Distribution of '+title)
plot.xlabel(title)
plot.ylabel('Frequency')
plot.legend()

# Show plot
plot.show()