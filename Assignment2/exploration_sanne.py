import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import math

# df_train = pd.read_csv("data/training_set_VU_DM.csv")

# print(df_train.columns)
#
# print(df_train.head())

"""
Category plot
"""
# categories = [138390, 83490, 4736470]
# plt.bar(["category 0", "category 1", "category 2"], categories)
# plt.ylabel("Number of rows")
# plt.show()


"""
Bezig gaan met de competitors

Fix missing values
Set all NULL values to 0

Combine
For rate: sum the rate of the competitors
For inv: set 0 if at least one of them is zero
For percentage: sum(rate * percentage) / rate
"""
df = pd.read_csv("data/training_short.csv")

COMP = 8
rates_col, invs_col, perc_col = [], [], []
for index, row in df.iterrows():
    rates, invs, percentages = [], [], []
    for i in range(COMP):
        rate = row[f"comp{i + 1}_rate"]
        inv = row[f"comp{i + 1}_inv"]
        percentage = row[f"comp{i + 1}_rate_percent_diff"]
        if math.isnan(rate):
            rate = 0
        if math.isnan(inv):
            inv = 0
        if math.isnan(percentage):
            percentage = 0
        else:
            percentage = rate * percentage
        rates.append(rate)
        invs.append(inv)
        percentages.append(percentage)

    percentage = sum(percentages)
    rate = sum(rates)

    # determine percentage based on rate
    if rate < 0:
        percentage /= - rate
    elif rate > 0:
        percentage /= rate

    if 0 in invs:
        inv = 0
    else:
        inv = 1

    rates_col.append(rate)
    invs_col.append(inv)
    perc_col.append(percentage)

comp_cols = []
for i in range(COMP):
    comp_cols.append("comp{i + 1}_rate")
    comp_cols.append("comp{i + 1}_inv")
    comp_cols.append("comp{i + 1}_rate_percent_diff")

df = df.drop(drop_cols, axis=1)
df["comp_rate"] = rates_col
df["comp_inv"] = invs_col
df["comp_perc"] = perc_col
