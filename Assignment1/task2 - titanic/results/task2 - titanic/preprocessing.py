import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


def importance(df_train):

    target = df_train['Survived'].values
    select_features = df_train.columns.values

    selector = SelectKBest(f_classif, len(select_features))
    selector.fit(df_train, target)
    scores = -np.log10(selector.pvalues_)
    indices = np.argsort(scores)[::-1]

    print('Features importance:')
    for i in range(len(scores)):
        print('%.2f %s' % (scores[indices[i]], select_features[indices[i]]))


def change_sex(df):
    """
    Change sex variable to binary (0 is male and 1 is female).
    """
    df["Sex"] = df.Sex.map({'male': 0, 'female': 1})

    return df


def is_alone(df):
    """
    Add column that states whether the passenger was travelling alone
    """

    df['alone'] = len(df) * [0]
    for index, row in df.iterrows():
        if row['SibSp'] == 0 and row['Parch'] == 0:
            df.at[index, 'alone'] = 1

    return df

def add_titles(data):
    """
    Extract titles from passengers' names and include them in new column.
    """
    titles = []
    for row in data["Name"]:
        new=re.split(r"[,.]+", row)
        titles.append(new[1].strip())

    # print(set(titles))

    # make new column with title category
    titles_cat = []
    for title in titles:
        if title in ["Don", "Sir", "Jonkheer", "the Countess", "Lady", "Dona"]:
            titles_cat.append("Noble")
        elif title in ["Ms", "Miss", "Mlle"]:
            titles_cat.append("Miss")
        elif title in ["Mrs", "Mme"]:
            titles_cat.append("Mrs")
        elif title in ["Capt", "Col", "Dr", "Major", "Rev"]:
            titles_cat.append("Other")
        elif title == "Mr":
            titles_cat.append(title)
        elif title == "Master":
            titles_cat.append(title)

    data["Title"] = titles_cat

    return data


def family_size(data):
    """
    Combine variable SibSp and Parch into new variable for family size.
    """
    data["FamSize"] = data["SibSp"] + data["Parch"]
    data.drop(["SibSp", "Parch"], axis=1, inplace=True)
    data['FamSize'] = data['FamSize'].apply(lambda x: "alone" if x == 0 else x if x < 4 else "4 or more")

    return data


def drop_uninteresting(data):
    """
    Drop variable that are not of interest.
    """
    uninteresting = ["Cabin", "Name", "Ticket"]
    data.drop(uninteresting, axis=1, inplace=True)

    return data


def scale(data):
    """
    Scale numerical values to values between -1 and 1.
    """
    to_scale = ["Age", "Fare"]
    scaler = StandardScaler()

    for var in to_scale:
        data[var] = data[var].astype("float64")
        data[var] = scaler.fit_transform(data[var].values.reshape(-1, 1))

    return data


def categorical(data):
    """
    Make categorical variables numerical by converting them to multiple binary
    variable columns for each factor in the variable.
    """

    # variables which need to be transformed to categorical
    to_categorical = ["Embarked", "Title", "FamSize", "Pclass"]

    for var in to_categorical:
        data = pd.concat([data, pd.get_dummies(data[var], prefix=var)], axis=1)
        del data[var]

    return data


def missing_values(data):
    """
    Replace missing values by a sensible value, i.e. the mean value of individuals in the same group.
    """

    # check for missing values: Age, Cabin, Embarked
    for col in data.columns.values:
        if data[col].isnull().any():
            print(f"Missing values in {col}")

    # replace missing age by mean of same title
    data["Age"].fillna(data.groupby("Title")["Age"].transform("mean"), inplace=True)

    # replace missing fare by mean of same class
    data["Fare"].fillna(data.groupby("Pclass")["Fare"].transform("mean"), inplace=True)

    # replace embarked by backfill method
    data["Embarked"].fillna(method="backfill", inplace=True)

    return data


def preprocess(df):
    """
    Preprocess the data set using all specified functions.
    """

    df = change_sex(df)
    df = is_alone(df)
    df = add_titles(df)
    df = family_size(df)

    df = missing_values(df)
    df = drop_uninteresting(df)
    df = categorical(df)
    df = scale(df)

    return df


def run_both():
    """
    Load and preprocess the training and testing datasets.
    """

    # load data
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")

    df_train = preprocess(df_train)
    df_test = preprocess(df_test)

    return df_train, df_test


if __name__ == "__main__":

    # get data
    df_train, df_test = run_both()

    # evaluate importance of all variables
    importance(df_train)

    # df_train, df_test = run_both()
    #
    # df_train.to_excel("train_processed.xlsx",sheet_name='clean')
    # df_test.to_excel("test_processed.xlsx",sheet_name='clean')
