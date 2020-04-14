import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import clean_all_Data
import math


df =  clean_all_Data.run_all(True)

# 45 entries are dropped
drop_list = []
for index, row in df.iterrows():
    if type(row['money']) == str or type(row['stress']) == str or type(row['gender']) == str:
        drop_list.append(index)

df = df.drop(drop_list)

options = {}
doubles = {}

# select values that occur multiple times (doubles)
for index, row in df.iterrows():

    # make dictionary with double values in the scatterplot
    coordinate = (row['money'], row['stress'])
    if coordinate in options.keys():
        if coordinate not in doubles.keys():
            doubles[coordinate] = [row['gender'], options[coordinate]]
        else:
            doubles[coordinate].append(row['gender'])

    # keep track of all coordinates
    options[coordinate] = row['gender']


# change doubles into average and amount of people
for i in doubles.keys():
    average = sum(doubles[i])/len(doubles[i])
    doubles[i] = [average, len(doubles[i])]

# set gender color
for index, row in df.iterrows():
    coordinate = (row['money'], row['stress'])
    if coordinate in doubles.keys():
        df.at[index, "gender"] = doubles[coordinate][0]

# add size to df
df['size'] = len(df) * [5]
for index, row in df.iterrows():
    coordinate = (row['money'], row['stress'])
    if coordinate in doubles.keys():
        df.at[index, 'size'] = (5 + doubles[coordinate][1] * 2)

df['money'] = pd.to_numeric(df['money'])
df['stress'] = pd.to_numeric(df['stress'])

ax = df.plot.scatter(x='money', y='stress', s=df['size'], c='gender', colormap='cool')
ax.plot()
ax.set_title("Money in relation to stresslevel for different gender", fontsize=14)
ax.set_ylabel("Stresslevel", fontsize=12)
ax.set_xlabel("Money", fontsize=12)
plt.show()
