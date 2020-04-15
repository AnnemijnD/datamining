import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler

def change_sex(df):
    for index, row in df.iterrows():
        if row["Sex"] == "male":
            df.at[index, "Sex"] = 0
        else:
            df.at[index, "Sex"] = 1

    return df


def add_titles(data):
    titles = []
    for row in data["Name"]:
        new=re.split(r"[,.]+", row)
        titles.append(new[1].strip())

    print(set(titles))

    # make new column with title category
    titles_cat = []
    for title in titles:
        if title in ["Don", "Sir", "Jonkheer"]:
            titles_cat.append("Noble male")
        elif title in ["the Countess", "Lady", "Dona"]:
            titles_cat.append("Noble female")
        elif title in ["Ms", "Miss", "Mlle"]:
            titles_cat.append("Miss")
        elif title in ["Mrs", "Mme"]:
            titles_cat.append("Mrs")
        elif title in ["Capt", "Col", "Dr", "Major", "Rev", "Master"]:
            titles_cat.append("Other")
        elif title == "Mr":
            titles_cat.append(title)

    data["Title"] = titles_cat

    return data


def drop_uninteresting(data):
    uninteresting = ["Cabin", "Name", "Ticket", "PassengerId"]
    data.drop(uninteresting, axis=1, inplace=True)

    return data


def scale(data):
    to_scale = ["Age", "Fare", "Parch", "Pclass", "SibSp"]
    scaler = StandardScaler()

    for var in to_scale:
        data[var] = data[var].astype("float64")
        data[var] = scaler.fit_transform(data[var].values.reshape(-1, 1))

    return data


def categorical(data):

    # variables which need to be transformed to categorical
    to_categorical = ["Embarked", "Title"]
    print(data.head())
    for var in to_categorical:
        data = pd.concat([data, pd.get_dummies(data[var], prefix=var)], axis=1)
        del data[var]

    return data

def missing_values(data):
    # check for missing values: Age, Cabin, Embarked
    for col in data.columns.values:
        if data[col].isnull().any():
            print(f"Missing values in {col}")

    # TODO: DIT IS OVERGENOMEN IK GA HIER NOG DINGEN VERANDEREN nuuu
    title_ages = dict(data.groupby("Title")["Age"].median())
    # create a column of the average ages
    data["age_med"] = data["Title"].apply(lambda x: title_ages[x])
    # replace all missing ages with the value in this column
    data["Age"].fillna(data["age_med"], inplace=True, )
    del data["age_med"]
    # impute missing Fare values using median of Pclass groups
    class_fares = dict(data.groupby("Pclass")["Fare"].median())
    # create a column of the average fares
    data["fare_med"] = data["Pclass"].apply(lambda x: class_fares[x])
    # replace all missing fares with the value in this column
    data["Fare"].fillna(data["fare_med"], inplace=True, )
    del data["fare_med"]
    data["Embarked"].fillna(method="backfill", inplace=True)

    return data

def run_all(df):

    df = change_sex(df)
    df = add_titles(df)
    df = drop_uninteresting(df)
    df = missing_values(df)
    df = categorical(df)
    df = scale(df)

    return df


def run_both():

    # load data
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")

    df_train = run_all(df_train)
    df_test = run_all(df_test)

    return df_train, df_test


if __name__ == "__main__":

    df_train, df_test = run_both()

    df_train.to_excel("train_processed.xlsx",sheet_name="clean")
    df_test.to_excel("test_processed.xlsx",sheet_name="clean")
