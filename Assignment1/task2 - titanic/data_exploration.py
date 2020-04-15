import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from preprocessing import add_titles, family_size

# load data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
pd.factorize(train_data["Survived"])

# preprocess data by adding titles and family sizes
train_data = add_titles(train_data)
train_data = family_size(train_data)


def plot(var):
    """
    Makes a plot for the surviving fraction of a variable of the training data set.
    """
    # MISSCHIEN KUNNEN WE HIER NOG IETS MEE
    # total_dead = len(train_data["Survived"] == 0)
    # total_survived = len(train_data["Survived"] == 1)
    # died = train_data[train_data["Survived"] == 0][var].value_counts() / total_dead
    # survived = train_data[train_data["Survived"] == 1][var].value_counts() / total_survived
    sns.set()
    sns.set_color_codes("pastel")

    # order bars for family size variable
    if var == "FamSize":
        sns.barplot(x=var, y="Survived", data=train_data, label="Total", color="b",\
                    capsize=.1, errwidth=.7, order=["alone", 1, 2, 3, "5 or more"])
    else:
        sns.barplot(x=var, y="Survived", data=train_data, label="Total", color="b",\
                    capsize=.1, errwidth=.7)

    plt.title("Ratio of survivors for variable " + str(var))
    plt.ylim([0, 1])
    plt.show()

# choose variables of interest
variables = ["Sex", "Title", "Embarked", "Pclass", "FamSize"]

# make plots for each variable
# for var in variables:
#     plot(var)
