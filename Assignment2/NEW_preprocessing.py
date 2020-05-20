import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import time
import random
import math
import time



def drop_cols(df, cols):
    """
    Drop variables that are not of interest.
    """
    df.drop(cols, axis=1, inplace=True)

    return df


def add_category(df):
    """
    Add category to training data.
    """
    df["category"] = df.apply(lambda row: transform_cat(row), axis=1)
    return df


def transform_cat(row):
    """
    Give category a value based on booked/clicked.
    """
    if row["booking_bool"] == 1:
       return 5
    if row["booking_bool"] == 0 and row["click_bool"] == 1:
       return 1
    else:
       return 0


def get_train_data(df):
    """
    Select 8% of the  data based on the categories.
    """

    srch_order = []
    cat0 = df[df.category == 5].index
    cat1 = df[df.category == 1].index
    cat2 = df[df.category == 0].index
    amount = int(len(df) * .04)
    print("amount of rows selected: ", amount)

    cat2_selec = np.random.choice(cat2, amount, replace=False)

    cat012 = np.concatenate((cat0, cat1, cat2_selec))

    df_selection = df.loc[cat012]

    return df_selection


def competitors(df):
    """
    Add all competitor variables to each other.
    """
    comp_vars = ["rate", "inv", "rate_percent_diff"]

    for var in comp_vars:
        comp_var = df[f"comp1_{var}"].fillna(0)

        for i in range(2, 9):
            comp_var += df[f"comp{i}_{var}"].fillna(0)

        for i in range(1, 9):
            df = drop_cols(df, [f"comp{i}_{var}"])

        df[f"comp_{var}"] = comp_var

        del var

    return df


def ranking(df):
    df['rank'] = df.groupby('srch_id')['category'].rank(ascending=True)
    print(df.head())
    return df


def missing_values(df):
    """
    EVEN CHECKEN OF NANS AL NIET MET 0 WORDEN OPGEVULD (denk voor rating)
    """

    # penalise missing review scores
    df["prop_review_score"].fillna(-1, inplace=True)
    df["prop_location_score1"].fillna(-1, inplace=True)
    df["prop_location_score2"].fillna(-1, inplace=True)
    df["visitor_hist_starrating"].fillna(-1, inplace=True)
    df["visitor_hist_adr_usd"].fillna(-1, inplace=True)

    # replace price by mean of hotels with same starrating
    mean_price_starrating = df.groupby("prop_starrating")["prop_log_historical_price"].transform("mean")
    df["prop_log_historical_price"].fillna(mean_price_starrating, inplace=True)

    # fill by worst possible value in dataset
    aff_min = df["srch_query_affinity_score"].min()
    df["srch_query_affinity_score"].fillna(aff_min, inplace=True)

    # TODO: is dit worst???? hoezo is verder weg slechter?
    orig_max = df["orig_destination_distance"].max()
    df["orig_destination_distance"].fillna(orig_max, inplace=True)

    # remaining mv's are replaced by mean of column
    # df = df.fillna(df.mean())
    print("er zijn nog zoveel nans: ", df.isnull().sum().sum())

    return df


def price_star_diff(df):
    """
    Calculate the absolute difference between respectively the starrating and
    price and the history of the user.
    If historical data is missing, the penalty is -1.
    """
    df["star_diff"] = abs(df["visitor_hist_starrating"] - df["prop_starrating"])
    no_hist = df[df.visitor_hist_starrating == -1].index
    df["star_diff"].loc[no_hist] = -1

    df["price_diff"] = abs(df["visitor_hist_adr_usd"] - df["price_usd"])
    no_hist = df[df.visitor_hist_adr_usd == -1].index
    df["price_diff"].loc[no_hist] = -1
    return df


def seasonality(df):
    """
    Save month of travel.
    """
    df_datetime = pd.DatetimeIndex(df.date_time)
    df["month"] = df_datetime.month

    return df


def scale(df):
    """
    Scale numerical values to values between -1 and 1.
    """

    scaler = StandardScaler()

    # select numerical variables
    vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # exclude prop_id, srch_id
    vars.remove("prop_id")
    vars.remove("srch_id")

    for var in vars:
        df[var] = df[var].astype("float64")
        df[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))

    return df


def feature_extraction(df):


    # DELETE THESE ROWS IF THE FUNCTION IS USED IN COMBINATION WITH FMV
    df["visitor_hist_starrating"].fillna(-1, inplace=True)
    df["visitor_hist_adr_usd"].fillna(-1, inplace=True)

    """ star diff: absolute diff, all rows with null values in hist are -1 """



    """ price diff: absolute diff, all rows with null values in hist are -1 """
    # get the absolute difference
    price_diff = abs(df["visitor_hist_adr_usd"] - df["price_usd"])

    # get the locations of the original null values
    no_hist = df[df.visitor_hist_adr_usd == -1].index
    price_diff.loc[no_hist] = -1

    # combine the two dfs --> add the column
    df = pd.concat([df, price_diff], axis=1)
    df = df.rename(columns={0: "price_diff"})

    """ book_prob """

    # booking(prop_id) / counting(prop_id)
    # number of times that prop_id was booked /number of times prop_id appeared in the data


    """ click_prob """


def prep_data(df, datatype):
    """
    Call all preprocessing functions for training and test set.
    """

    start = time.time()
    if datatype == "training":
        df = drop_cols(df, "gross_bookings_usd")
        df = add_category(df)
        df = get_train_data(df)
        df = ranking(df)
    print("(1/6 - train only) add categories and downsample train data: ", np.round((time.time() - start)*1000 / 60, 2), "min")

    start = time.time()
    df = competitors(df)
    print("(2/6) competitors: ", np.round((time.time() - start)*1000 / 60, 2), "min")

    start = time.time()
    df = seasonality(df)
    print("(3/6) seasons: ", np.round((time.time() - start)*1000 / 60, 2), "min")

    start = time.time()
    df = missing_values(df)
    print("(4/6) missing values: ", np.round((time.time() - start)*1000 / 60, 2), "min")

    start = time.time()
    df = price_star_diff(df)
    print("(5/6) price and star difference: ", np.round((time.time() - start)*1000 / 60, 2), "min")

    start = time.time()
    df = scale(df)
    print("(6/6) scaling: ", np.round((time.time() - start)*1000 / 60, 2), "min")

    return df


if __name__ == "__main__":
    """
    Run this file to preprocessing training and test data.
    Specify the output files accordingly.

    Preprocessing includes:
        * Dropping irrelevant columns
        * Add category for clicking/booking
        * Downsample data based on categories
        * Combine competitor features
        * (TODO:) add date time feature
        * (TODO:) add price and star diff feature
        * (TODO:) normalise log price
        * Fill missing values
        * Scale numerical variables

    """

    print("\nSTART PREPROCESSING DATA\n")

    # datatypes = ["training", "test"]

    # datatypes = ["test"]
    datatypes = ["training"]

    for datatype in datatypes:

        save_filepath = f"data/{datatype}_prep_newTEST.csv"

        start = time.time()
        # open_filepath = f"data/{datatype}_set_VU_DM.csv"
        open_filepath = f"data/{datatype}_short.csv"


        print(f"\n\nset {datatype.upper()} from file {open_filepath} preprocessed and saved in {save_filepath}\n")

        # open files
        df = pd.read_csv(open_filepath)
        print("file loading: ", (time.time() - start)*1000 / 60, "min")

        # preprocess data
        df = prep_data(df, datatype)

        # save preprocessed data
        print("\ntotal time: ", np.round((time.time() - start)*1000 / 60, 2))
        df.to_csv(save_filepath)

        del df
