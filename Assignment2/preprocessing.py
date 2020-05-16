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


def add_category(df):
    """
    Add a category based on whether it is booked and clicked, only clicked or neither
    Only need to run this function once!
    """
    categories = []
    for index, row in df.iterrows():
        booked = row['booking_bool']
        clicked = row['click_bool']
        if booked:
            category = 5
        elif clicked:
            category = 1
        else:
            category = 0
        categories.append(category)
    df["category"] = categories
    df.to_csv("data/train_category.csv", index=False)

    """
    If test data needs extra row: comment above and uncomment below
    """
    # df["category"] = [0] * len(df)
    # df.to_csv("data/train_category.csv", index=False)


def get_train_data(df):
    """
    Select 8% of the  data based on the categories.
    Only need to run this function once!
    """

    cat0 = df[df.category == 5].index
    cat1 = df[df.category == 1].index
    cat2 = df[df.category == 0].index
    amount = int(len(df) * .04)
    print("amount of rows selected: ", amount)
    cat2_selec = np.random.choice(cat2, amount, replace=False)

    cat012 = np.concatenate((cat0, cat1, cat2_selec))

    df_selection = df.loc[cat012]

    df_selection.to_csv("data/train_selection.csv", index=False)


def scale(data, vars):
    """
    Scale numerical values to values between -1 and 1.
    """
    scaler = StandardScaler()

    for var in vars:
        if var not in ["prop_id","srch_id"]:
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
    df_train = add_category(df_train)
    df_train = get_train_data(df_train)
    uninteresting = ["srch_adults_count", "srch_children_count", "srch_room_count", "date_time", "site_id"]
    df_test = drop_cols(df_test, uninteresting)
    numeric_train, categorical_train = overview(df_train)
    numeric_test, categorical_test = overview(df_test)
    df_train = missing_values(df_train)
    df_test = missing_values(df_test)
    df_train = scale(df_train, numeric_train)
    df_test = scale(df_test, numeric_test)

    return df_train, df_test


def combine_competitors(df):
    """
    Set all NULL values to 0

    Combine
    For rate: sum the rate of the competitors
    For inv: set 0 if at least one of them is zero
    For percentage: sum(rate * percentage) / rate
        (if rate is zero then don't divide)
    """
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

    return df

if __name__ == "__main__":
    """
    RUN THIS FILE ONCE FOR train_selection AND FOR test_category
    WHEN FUNCTIONS ARE SPECIFIC FOR TRAIN OR TEST SPECIFY THIS!
    After that the preprocessed data will be saved in "preprocessed_train.csv"
    Make sure to delete the previous preprocessed file
    """

    """ Select train or test """
    clean = "train"
    # clean = "test"

    """ load data you want to preprocess """
    if clean == "train":
        # df = pd.read_csv("data/train_selection.csv")
        df = pd.read_csv("data/training_short.csv")
    else:
        # df = pd.read_csv("data/test_category.csv")
        df = pd.read_csv("data/test_short.csv")

    """ Combine competitor cols """
    df = combine_competitors(df)


    """ drop cols TODO: CHECK OF DIT ZO IS VOOR TRAIN EN TEST"""
    # uninteresting = ["srch_adults_count", "srch_children_count", "srch_room_count", "date_time", "site_id", "gross_bookings_usd"]
    # data = drop_cols(df, uninteresting)


    """ TODO: make cols categorical """
    # data["Pclass"] = pd.Categorical(data.Pclass)


    """ overview of numerical and categorical data """
    # numeric, categorical = overview(data)

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


    """ Save data in a csv file """
    df.to_csv(f"data/preprocessed_{clean}.csv", index=False)
