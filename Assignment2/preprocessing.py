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
import math
from numba import jit # does not work with pandas
import time


def display_df(df):
    """
    Display full dataframe in terminal.
    """
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 100):
        display(df)


def importance():
    """
    Optional: importance estimation.
    """
    # memoryerror for large dataset
    target = data['booking_bool'].values
    select_features = data.columns.values

    selector = SelectKBest(f_classif, len(select_features))
    selector.fit(data, target)
    scores = -np.log10(selector.pvalues_)
    indices = np.argsort(scores)[::-1]

    print('Features importance:')
    for i in range(len(scores)):
        print('%.2f %s' % (scores[indices[i]], select_features[indices[i]]))

    # most important: click_bool > position > random_bool > prop_location_score2


def shorten():
    """
    Shorten large dataset to only 1000 rows for facilitating inspection of data.
    """

    # load data
    df_train = pd.read_csv("data/training_set_VU_DM.csv")
    # df_train = pd.read_csv("data/train_selection.csv")
    # df_test = pd.read_csv("data/test_set_VU_DM.csv")
    # df_train = pd.read_csv("data/train_prep_long.csv")
    # df_test = pd.read_csv("data/test_prep_long.csv")
    # df = pd.read_csv("results/solutions/xgboost_2020-05-17-21-02.csv")
    # print(df_train.head(10))
    # print(df_test.head(10))

    df_train.sample(n=1000).to_csv("data/training_short.csv", index=False)
    # df_test.sample(n=1000).to_csv("data/test_short.csv", index=False)
    # df.sample(n=1000).to_csv("data/xg_short.csv", index=False)

def overview(data):
    """
    Overview of the data, returns numeric and categorical variables in list.
    """
    # summaries of categorical and numerical data
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
    # print('numeric columns: ' + str(numeric_columns))
    # display_df(round(data[numeric_columns].describe(),2)) # MEMORYERROR
    # print('categorical columns: ' + str(categorical_columns))
    # print(data[categorical_columns].describe())

    return numeric_columns, categorical_columns


# def add_category(df):
#     """
#     Add a category based on whether it is booked and clicked, only clicked or neither
#     Only need to run this function once!
#     """
#     categories = []
#     for index, row in df.iterrows():
#         booked = row['booking_bool']
#         clicked = row['click_bool']
#         if booked:
#             category = 5
#         elif clicked:
#             category = 1
#         else:
#             category = 0
#         categories.append(category)
#     df["category"] = categories
#
#     return df


def add_category2(df):
    booked = df[df.booking_bool == 1].index
    clicked = df[df.click_bool == 1].index

    nr_rows = len(df)
    categories = []
    for row in range(nr_rows):
        if row in booked:
            categories.append(5)
        elif row in clicked:
            categories.append(1)
        else:
            categories.append(0)

    df["category"] = categories

    return df


def add_searchorder(df):
    """
    Adds columns (number of items in search, number of times hotel  is booked,
                    rank of item in search)
        Args:
            df (pandas dataframe)

        Returns
            df (pandas dataframe )
    """
    # add column n_search n_srchitems

    # df = pd.read_csv("data/test_sohet_VU_DM.csv")

    df['n_srchitems'] = df.groupby('srch_id')['srch_id'].transform('count')

    # add column n_booked
    df['n_booked'] = df.groupby('prop_id')['prop_id'].transform('count')

    # add collumn srch_rank
    df["srch_rank"] = df.groupby("srch_id")["srch_id"].rank("first", ascending=True)

    # print(df.shape)

    return df


def get_train_data(df):
    """
    Select 8% of the  data based on the categories.
    Only need to run this function once!
    """

    srch_order = []
    cat0 = df[df.category == 5].index
    cat1 = df[df.category == 1].index
    cat2 = df[df.category == 0].index
    amount = int(len(df) * .04)
    # print("amount of rows selected: ", amount)

    cat2_selec = np.random.choice(cat2, amount, replace=False)

    cat012 = np.concatenate((cat0, cat1, cat2_selec))

    df_selection = df.loc[cat012]

    return df_selection


def scale(data, vars):
    """
    Scale numerical values to values between -1 and 1.
    """
    scaler = StandardScaler()

    for var in vars:
        data[var] = data[var].astype("float64")
        data[var] = scaler.fit_transform(data[var].values.reshape(-1, 1))

    return data


def transform(data, vars):
    """
    TODO: transform categorical cols.
    """
    # data["Pclass"] = pd.Categorical(data.Pclass)


def missing_values(data):
    """
    Replace missing values by a sensible value, i.e. the mean value of individuals in the same group.
    """
    # print("MISSING VALS BEFORE")
    # print(data.isnull().sum())

    # TODO: fillnas with e.g. mean value of comparable data group
    # for var in numeric_columns:
    #     data[var].fillna(data.groupby("booking_bool")[var].transform("mean"), inplace=True)

    # now data is filled with the mean of the col
    data = data.fillna(data.mean())

    # print("MISSING VALS AFTER")
    # print(data.isnull().sum())


    return data


def fill_missing_val(df):
    """
    Fill the missing values per column
    """

    """
    prop_review_score
    change nan to -1
    """
    df["prop_review_score"].fillna(-1, inplace=True)

    """
    prop_location_score2
    change nan to -1
    """
    df["prop_location_score2"].fillna(-1, inplace=True)

    """
    srch_query_affinity_score

    GEEN IDEE NOG OF DIT HANDIG IS..... ZE HEBBEN VAST MET EEN REDEN DE LOG GENOMEN?
    Zet ze nu weer om naar probabilities tussen 0 en 1, maar denk dat zij het hadden omgezet omdat
    je mega kleine waarden krijgt.... Weet niet hoe kut dat is?
    Iig ook de null waarden naar -1 gezet nu.
    """
    # nan_rows = df[df.srch_query_affinity_score.isnull()].index
    #
    # rows = range(len(df))
    # float_rows = sorted(list(set(rows) - set(nan_rows)))
    # for i in float_rows:
    #     df.at[i, "srch_query_affinity_score"] = math.exp(df["srch_query_affinity_score"].iloc[i])
    #
    #     # als met exp niet werkt is dit een andere optie
    #     # df.at[i, "srch_query_affinity_score"] = -1 / df["srch_query_affinity_score"].iloc[i]
    #
    #     # als dat ook niet werkt zouden we als laatste optie alle nan waarden de
    #     # minimum waarde van de kolom kunnen geven
    #
    # for i in nan_rows:
    #     df.at[i, "srch_query_affinity_score"] = -1

    # ============== above is the 'original'
    # BOVENSTAANDE NIET VERWIJDEREN NOG. NU SOWIESO IETS WAT NIET ERRORT/ LANG DUURT
    # MAAR BOVENSTAANDE IDEE IS MISS SLIMMER.



    df["srch_query_affinity_score"].fillna(1, inplace=True)
    min_val = min(df["srch_query_affinity_score"])

    df.loc[df["srch_query_affinity_score"] == 1, ["srch_query_affinity_score"]] = min_val


    # print(df["srch_query_affinity_score"])


    """
    orig_destination_distance
    change nan to -1
    """
    df["orig_destination_distance"].fillna(-1, inplace=True)

    return df


def drop_cols(df, uninteresting):
    """
    Drop variables that are not of interest.
    """
    df.drop(uninteresting, axis=1, inplace=True)

    return df
#
# def combine_competitors(df):
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
        comp_cols.append(f"comp{i + 1}_rate")
        comp_cols.append(f"comp{i + 1}_inv")
        comp_cols.append(f"comp{i + 1}_rate_percent_diff")

    df = df.drop(comp_cols, axis=1)
    df["comp_rate"] = rates_col
    df["comp_inv"] = invs_col
    df["comp_perc"] = perc_col

    return df

def combine_competitors2(df):

    # get all competitor column names
    COMP = 8
    comp_cols = []
    for i in range(COMP):
        comp_cols.append(f"comp{i + 1}_rate")
        comp_cols.append(f"comp{i + 1}_inv")
        comp_cols.append(f"comp{i + 1}_rate_percent_diff")

    # fill NULL values with zero
    for column in comp_cols:
        df[column].fillna(0, inplace=True)



    rates_col, invs_col, perc_col = [], [], []
    for index, row in df.iterrows():
        rates, invs, percentages = [], [], []
        for i in range(COMP):
            rate = row[f"comp{i + 1}_rate"]
            inv = row[f"comp{i + 1}_inv"]
            percentage = row[f"comp{i + 1}_rate_percent_diff"]
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

    df = df.drop(comp_cols, axis=1)
    df["comp_rate"] = rates_col
    df["comp_inv"] = invs_col
    df["comp_perc"] = perc_col

    return df

#
def prep_data(df, settype):
    """
    Call all preprocessing functions for training and test set.
    """
    start = time.time()


    if settype == "train":
        df = add_category2(df)
        df = get_train_data(df)
        print("door add categories:", time.time() - start)


    df = combine_competitors2(df)
    print("door de eerste combine_competitors:",time.time() - start)
    exit()

    uninteresting = ["srch_adults_count", "srch_children_count", "srch_room_count", "date_time", "site_id", "gross_bookings_usd"]
    df = drop_cols(df, uninteresting)
    print("door drop cols:", time.time() - start)


    df = add_searchorder(df)
    print("door de eerste search order:", time.time() - start)

    # df = missing_values(df)

    df = fill_missing_val(df)
    print("door de missing values:", time.time() - start)

    numeric, categorical = overview(df)
    print(numeric)
    print("door beide numeric en categorial tests:", time.time() - start)

    # avoid scaling of boolean variables and important id's
    for boolean in ['random_bool', "prop_brand_bool", "promotion_flag", 'srch_saturday_night_bool', "srch_id", "prop_id"]:
        numeric.remove(boolean)
        # numeric_test.remove(boolean)

    print("door de boolean removal:",time.time() - start)

    df = scale(df, numeric)
    print("door de scaling:", time.time() - start)

    return df

if __name__ == "__main__":

    """
    RUN THIS FILE ONCE FOR train_selection AND FOR test_category
    WHEN FUNCTIONS ARE SPECIFIC FOR TRAIN OR TEST SPECIFY THIS!
    After that the preprocessed data will be saved in "preprocessed_train.csv"
    Make sure to delete the previous preprocessed file

    TO AVOID MEMORY ERROR: run first 1 large and one small file, save the output
    of the large file, then switch and run again
    """

    settype = None
    while not(settype == "train" or settype == "test"):
        # settype = input("train or test:").lower()
        settype = "test"

    save_filepath = f"data/{settype}_preprocessed.csv"
    open_filepath = "data/test_set_VU_DM.csv"
    print("HOI LEES JE DIT WEL LEZEN HE!!!!\n")
    print(f"set {settype.upper()} from file {open_filepath} preprocessed and saved in {save_filepath}\n")

    # open files
    df = pd.read_csv(open_filepath)

    df = prep_data(df, settype)

    # df.to_csv(save_filepath)
