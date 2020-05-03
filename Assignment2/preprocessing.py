import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from IPython.display import display
import time
from tqdm import tqdm


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


def drop_cols(df):
    """
    Drop variables that are not of interest.
    """
    uninteresting = ["srch_adults_count", "srch_children_count", "srch_room_count", "date_time", "site_id", "gross_bookings_usd"]
    df.drop(uninteresting, axis=1, inplace=True)

    return df


if __name__ == "__main__":

    """ load data """
    df_train = pd.read_csv("data/training_short.csv")
    # df_test = pd.read_csv("data/test_short.csv")


    """ drop cols """
    data = drop_cols(df_train)


    """ TODO: make cols categorical """


    """ overview of numerical and categorical data """
    # summaries of categorical and numerical data
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
    print('numeric columns: ' + str(numeric_columns))
    display_df(round(data[numeric_columns].describe(),2)) # MEMORYERROR
    print('categorical columns: ' + str(categorical_columns))
    # print(data[categorical_columns].describe())


    """ missing values """
    print("MISSING VALS BEFORE")
    for col in data.columns.values:
        if data[col].isnull().any():
            print(f"Missing values in {col}")

    # TODO: fillnas with mean value of comparable data group
    # for var in numeric_columns:
    #     data[var].fillna(data.groupby("booking_bool")[var].transform("mean"), inplace=True)

    # now data is filled with the mean of the col
    data = data.fillna(data.mean())

    print("MISSING VALS AFTER")
    for col in data.columns.values:
        if data[col].isnull().any():
            print(f"Missing values in {col}")


    """ TODO: combine competitor cols """
    # now mean is taken of comp_rates, do we want to make 1 competition score based on
    # 3 available competitor variables (rates inv diff)?
    data["comprate"] = data.loc[:,['comp1_rate','comp2_rate','comp3_rate','comp4_rate',\
                        'comp5_rate','comp6_rate','comp7_rate','comp8_rate']].mean(axis=1)
    print(data.comprate[:100])


    """ scaling numeric cols"""
    scaler = StandardScaler()

    for var in numeric_columns:
        data[var] = data[var].astype("float64")
        data[var] = scaler.fit_transform(data[var].values.reshape(-1, 1))

    data.to_csv("data/training_scaled.csv", index=False)


    """ TODO: transform categorical variables """


    """ optional: importance estimation """

    # multiple targets?
    target = data['position'].values
    select_features = data.columns.values

    selector = SelectKBest(f_classif, len(select_features))
    selector.fit(data, target)
    scores = -np.log10(selector.pvalues_)
    indices = np.argsort(scores)[::-1]

    print('Features importance:')
    for i in range(len(scores)):
        print('%.2f %s' % (scores[indices[i]], select_features[indices[i]]))
