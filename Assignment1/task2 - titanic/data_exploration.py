import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from preprocessing import add_titles, family_size
from IPython.display import display

# load data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
data = pd.concat([train_data, test_data], axis=0, sort=True)

# get latex table for summaries of categorical and numerical data
train_data["Pclass"] = pd.Categorical(train_data.Pclass)
numeric_columns = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = train_data.select_dtypes(include=["object", "category"]).columns.tolist()
print('numeric columns: ' + str(numeric_columns))
print(round(train_data[numeric_columns].describe(),2).to_latex())
print('categorical columns: ' + str(categorical_columns))
print(train_data[categorical_columns].describe().to_latex())

# preprocess data by adding titles and family sizes
train_data = add_titles(train_data)
train_data = family_size(train_data)

# to view dataset
def display_df(df):
    """
    Display full dataframe in terminal.
    """
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 100):
        display(df)


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
                    capsize=.1, errwidth=.7, order=["alone", 1, 2, 3, "4 or more"])
    else:
        sns.barplot(x=var, y="Survived", data=train_data, label="Total", color="b",\
                    capsize=.1, errwidth=.7)

    plt.title("Ratio of survivors for variable " + str(var), fontsize=16)
    plt.ylim([0, 1])
    plt.savefig("results/survived_" + str(var) + ".png")
    plt.show()




# choose variables of interest based on overview of data
display_df(train_data.describe(include='all').T)
variables = ["Sex", "Title", "Embarked", "Pclass", "FamSize"]

# make plots for each variable
for var in variables:
    plot(var)
