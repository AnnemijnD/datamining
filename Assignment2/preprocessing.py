import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from IPython.display import display
import time
from tqdm import tqdm
import random


def display_df(df):
    """
    Display full dataframe in terminal.
    """
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 100):
        display(df)


def shorten():
    """
    Shorten large dataset to only 1000 rows for facilitating inspection of data.
    """

    # load data
    df_train = pd.read_csv("data/training_set_VU_DM.csv")
    df_test = pd.read_csv("data/test_set_VU_DM.csv")

    print(df_train.head(10))
    print(df_test.head(10))

    df_train.sample(n=1000).to_csv("data/training_short.csv", index=False)
    df_test.sample(n=1000).to_csv("data/test_short.csv", index=False)


def overview(data):
    """
    Overview of the data, returns numeric and categorical variables in list.
    """
    # summaries of categorical and numerical data
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
    print('numeric columns: ' + str(numeric_columns))
    display_df(round(data[numeric_columns].describe(),2)) # MEMORYERROR
    print('categorical columns: ' + str(categorical_columns))
    # print(data[categorical_columns].describe())

    return numeric_columns, categorical_columns


def scale(data, vars):
    """
    Scale numerical values to values between -1 and 1.
    """
    scaler = StandardScaler()

    for var in vars:
        data[var] = data[var].astype("float64")
        data[var] = scaler.fit_transform(data[var].values.reshape(-1, 1))

    return data


def missing_values(data):
    """
    Replace missing values by a sensible value, i.e. the mean value of individuals in the same group.
    """
    print("MISSING VALS BEFORE")
    # for col in data.columns.values:
    #     if data[col].isnull().any():
    #         print(f"Missing values in {col}")
    print(data.isnull().sum())
    # TODO: fillnas with e.g. mean value of comparable data group
    # for var in numeric_columns:
    #     data[var].fillna(data.groupby("booking_bool")[var].transform("mean"), inplace=True)

    # now data is filled with the mean of the col
    data = data.fillna(data.mean())

    print("MISSING VALS AFTER")
    print(data.isnull().sum())
    # for col in data.columns.values:
    #     if data[col].isnull().any():
    #         print(f"Missing values in {col}")

    return data


def drop_cols(df, uninteresting):
    """
    Drop variables that are not of interest.
    """
    df.drop(uninteresting, axis=1, inplace=True)

    return df


def prep_data(df_train, df_test):
    uninteresting = ["srch_adults_count", "srch_children_count", "srch_room_count", "date_time", "site_id", "gross_bookings_usd"]
    df_train = drop_cols(df_train, uninteresting)
    uninteresting = ["srch_adults_count", "srch_children_count", "srch_room_count", "date_time", "site_id"]
    df_test = drop_cols(df_test, uninteresting)
    numeric_train, categorical_train = overview(df_train)
    numeric_test, categorical_test = overview(df_test)
    df_train = missing_values(df_train)
    df_test = missing_values(df_test)
    df_train = scale(df_train, numeric_train)
    df_test = scale(df_test, numeric_test)

    return df_train, df_test

def add_category(df):
    """
    Add a category based on whether it is booked and clicked, only clicked or neither
    """
    # categories = []
    # for index, row in df.iterrows():
    #     booked = row['booking_bool']
    #     clicked = row['click_bool']
    #     if booked:
    #         category = 0
    #     elif clicked:
    #         category = 1
    #     else:
    #         category = 2
    #     categories.append(category)
    # df["category"] = categories
    df["category"] = [0] * len(df)

    df.to_csv("data/test_category.csv")


def get_train_data():
    """
    Select 8% of the  data based on the categories
    """
    df = pd.read_csv("data/train_category.csv")

    cat0 = df[df.category == 0].index
    cat1 = df[df.category == 1].index
    cat2 = df[df.category == 2].index
    cat2_selec = np.random.choice(cat2, 223125, replace=False)

    cat012 = np.concatenate((cat0, cat1, cat2_selec))

    df_selection = df.loc[cat012]

    df_selection.to_csv("data/train_selection.csv")

    print(len(df_selection))

if __name__ == "__main__":

    """ load data """
    # df_train = pd.read_csv("data/training_set_VU_DM.csv")
    # df_train = pd.read_csv("data/training_short.csv")
    # df_train = pd.read_csv("data/training_set_VU_DM.csv")
    # df_test = pd.read_csv("data/test_short.csv")
    df_test = pd.read_csv("data/test_set_VU_DM.csv")

    """ add category column """
    add_category(df_test)

    # df_train.to_csv("data/train_category.csv")
    # get_train_data()

    pass

    """ drop cols """
    data = drop_cols(df_train)


    """ TODO: make cols categorical """
    # zijn er uberhaupt categorische variabelen?
    # data["Pclass"] = pd.Categorical(data.Pclass)


    """ overview of numerical and categorical data """
    numeric, categorical = overview(data)


    """ missing values (TODO) """
    data = missing_values(data)


    """ TODO: combine competitor cols """
    # now mean is taken of comp_rates, do we want to make 1 competition score based on
    # 3 available competitor variables (rates inv diff)?
    data["comprate"] = data.loc[:,['comp1_rate','comp2_rate','comp3_rate','comp4_rate',\
                        'comp5_rate','comp6_rate','comp7_rate','comp8_rate']].mean(axis=1)


    """ scaling numeric cols"""
    data = scale(data, numeric)
    # data.to_csv("data/training_preprocessed.csv")


    """ TODO: transform categorical variables """
    # nvt als er geen categorische variabelen zijn


    """ optional: importance estimation """
    # memoryerror for large dataset
    # target = data['booking_bool'].values
    # select_features = data.columns.values
    #
    # selector = SelectKBest(f_classif, len(select_features))
    # selector.fit(data, target)
    # scores = -np.log10(selector.pvalues_)
    # indices = np.argsort(scores)[::-1]
    #
    # print('Features importance:')
    # for i in range(len(scores)):
    #     print('%.2f %s' % (scores[indices[i]], select_features[indices[i]]))

    # most important: click_bool > position > random_bool > prop_location_score2
