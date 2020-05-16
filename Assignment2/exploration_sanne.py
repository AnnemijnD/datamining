import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt

# df_train = pd.read_csv("data/training_set_VU_DM.csv")

# print(df_train.columns)
#
# print(df_train.head())

# category plot
categories = [138390, 83490, 4736470]
plt.bar(["category 0", "category 1", "category 2"], categories)
plt.ylabel("Number of rows")
plt.show()
