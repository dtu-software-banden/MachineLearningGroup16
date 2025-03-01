# exercise 3.1.1
import importlib_resources
import numpy as np
import xlrd
import pandas as pd

# Load xls sheet with data
filename = "C:/Users/katin/OneDrive - Danmarks Tekniske Universitet/6_semester/MachineLearning/autoMPG/auto-mpg-revised-ml.data.csv"
data = pd.read_csv(filename, delimiter='\t')

# Extract attribute names (1st row, column 4 to 12)
attributeNames = data.columns[0:6].tolist()
print(attributeNames)


# creating a Dataframe object 
df = pd.DataFrame(data)

# Z-Score using pandas standerizse
#for i in attributeNames:
#    df[i] = (df[i] - df[i].mean()) / df[i].std() 

# Extract class names to python list,
# then encode with integers (dict)
classLabels = df.iloc[0:393, -2].tolist()
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(classNames))))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])
print(y)

# Preallocate memory, then extract excel data to matrix X
X = df.iloc[0:393, 0:6].to_numpy()
print(X)

#for i, col_id in enumerate(range(0, 8)):
    #X[:, i] = np.asarray(data)

# Compute values of N, M and C.
#N = len(y)

N = X.shape[0]
M = len(attributeNames)
C = len(classNames)

print("Ran Exercise 3.1.1")
