import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# x = np.arange(-3, 3, 0.1)
# y = np.sin(x)
# plt.plot(x, y)
# plt.show()
df = pd.read_csv('./input/train.csv')
print(df.dtypes)
print(df.head(3))
print(df.describe())
