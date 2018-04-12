import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('./input/train.csv')
# print(df.dtypes)
# print(df.head(3))
# print(df.describe())

# Cleaning
df= pd.read_csv('./input/train.csv').replace('male', 0).replace('female', 1)

# Visualize
split_data = []
for survived in [0, 1]:
    split_data.append(df[df.Survived == survived])

temp = [i["Pclass"].dropna() for i in split_data]
plt.hist(temp, histtype="barstacked", bins=3)
plt.show()

temp = [i["Age"].dropna() for i in split_data]
plt.hist(temp, histtype="barstacked", bins=16)
plt.show()
