# exercise 3.1.1
import importlib_resources
import numpy as np
import xlrd
import pandas as pd

# Load data into a pandas dataframe
filename = "auto-mpg-revised-ml.data.csv"
data = pd.read_csv(filename, delimiter='\t')
df = pd.DataFrame(data)


# Extract attribute names
# nly the fist 6 Attributes are taken into account
# as the last three are model year, origin and car name
attributeNames = data.columns[0:6].tolist()
print(attributeNames)


# Z-Score for standardizing te data
# for i in attributeNames:
#    df[i] = (df[i] - df[i].mean()) / df[i].std() 


# Extract class names to python list,
# then encode with integers (dict)
classLabels = df.iloc[0:393, -2].tolist()
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(classNames))))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Extract the 6 first columns into a matrix
X = df.iloc[0:393, 0:6].to_numpy()

N = X.shape[0]
M = len(attributeNames)
C = len(classNames)

