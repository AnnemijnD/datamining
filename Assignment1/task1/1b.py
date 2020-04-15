from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import clean_all_Data
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def CV(df):
    train_overall, test_overall = train_test_split(df)

    kfold = KFold(10, True)

    # train and test are arrays with the row indices from the train_overall set
    for train, test in kfold.split(train_overall):
        # print(train_overall[train])
        # print(train_overall[test])
        print("train: ", train)
        print("test: ", test)
        print(type(train))

        break
def classify(df):

    return

if __name__ == "__main__":
    df = clean_all_Data.run_all(False)
    CV(df)
    df = df.replace('NaN', np.nan)
    df = df.dropna(subset=["machinelearning", "informationretrieval", "statistics", "databases"])
    df = df.reset_index(drop=True)
    df2 = df["machinelearning"]
    df1 = df[["informationretrieval","statistics","databases"]]
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(df1, df2)

    correct = 0
    incorrect = 0
    for index, row in df.iterrows():
        ir = row["informationretrieval"]
        stats = row["statistics"]
        databases = row["databases"]
        gender = row["gender"]
        answer = row["machinelearning"]
        prediction = neigh.predict([[ir, stats, databases]])

        if prediction == answer:
            correct +=1
        else:
            # print(answer, prediction)
            incorrect += 1

    print(correct*100/(incorrect+correct))
