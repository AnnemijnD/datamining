from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import clean_all_Data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn import svm
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt




def CV(df):
    train_overall, test_overall = train_test_split(df)

    kfold = KFold(10, True)

    # train and test are arrays with the row indices from the train_overall set
    for train, test in kfold.split(train_overall):

        # print("train: ", train)
        # print("test: ", test)
        # print(type(train))

        break
def classify_KNN(X, y):

    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X, y)
    return neigh

def classify_svm(X, y):
    model = svm.SVC()
    model.fit(X, y)
    return model

def feature_selection(df1, X2, y, y2):


    # print(X.to_string())
    # gets the gains vector
    # X.to_excel("TEST.xlsx",sheet_name='clean')

    gain_vec = mutual_info_classif(X, y, discrete_features=True)

    delete_ind = gain_vec.argsort()[::-1][5:]
    delete = []
    for i in range(len(delete_ind)):
        delete.append(df1.columns[i])

    # deletes the features that can be deleted
    X_fil = df1.drop(delete, 1)

    X_fil2 = X2.drop(delete, 1)


    # print(X_fil)
    return X_fil, y, X_fil2

def predictions(model, X, y):

    correct = 0
    incorrect = 0
    for index, row in X.iterrows():
        # print(row)
        # exit(
        prediction = model.predict([row])

            # print(X)
        answer = y.loc[index]
        if prediction == answer:
            correct +=1
        else:
            incorrect += 1

    accuracy = correct*100/(incorrect+correct)
    return accuracy

if __name__ == "__main__":
    df = clean_all_Data.run_all(False)
    # est = preprocessing.KBinsDiscretizer(n_bins=4, encode='onehot', strategy='uniform')
    df["lateness_bedtime"] = pd.qcut(df['lateness_bedtime'], 4, labels=False)
    df2 = pd.get_dummies(df["lateness_bedtime"],prefix='lateness_bedtime')
    # df = df.drop("lateness_bedtime", axis=1)
    df = pd.concat([df, df2], axis=1)


    # x = df.values #returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # df = pd.DataFrame(x_scaled)
    df = df.dropna()
    df = df.reset_index(drop=True)
    train_overall, test_overall = train_test_split(df)

    kfold = KFold(10, True)

    chocolatelist = ["fat", "slim", "neither", "nan", "I have no idea what you are talking about"]
    bedtimelist = ["0.0", "1.0", "2.0", "3.0"]
    programmelist = ["CLS", "AI", "BA", "CS", "BF", "econometrics", "QRM", "other"]
    prefix = ["chocolate_", "lateness_bedtime_", "programme_"]

    corlists = [chocolatelist, bedtimelist, programmelist]

    for list in range(len(corlists)):
        for el in range(len(corlists[list])):
            corlists[list][el] = prefix[list] + corlists[list][el]

    data = []
    labels = []

    for col in df.columns:
        if col in chocolatelist or col in bedtimelist or col in programmelist:
            continue

        col_data = []
        labels.append(col)
        df.reset_index
        print(col.upper())
        # train and test are arrays with the row indices from the train_overall set
        for train, test in kfold.split(train_overall):
            df1 = df.drop(test.tolist(), axis=0)
            df2 = df.drop(train.tolist(), axis=0)
            df1 = df1.reset_index()
            df2 = df2.reset_index()
            # try:
            y_col = col

            # exit()
            # print(df1.head)
            if y_col[0:9] == "chocolate":
                # print(y_col)
                remove_list = [n for n in chocolatelist if n !=y_col]
                y = df1["chocolate"]
                df1 = df1.drop(remove_list, axis=1)

                y2 = df2["chocolate"]
                df2 = df2.drop(remove_list, axis=1)


                # print(df1.head)
            elif y_col[0:8] == "lateness":
                remove_list = [n for n in bedtimelist if n != y_col]
                df1 = df1.drop(remove_list, axis=1)
                df2 = df2.drop(remove_list, axis=1)
                y = df1["lateness_bedtime"]
                y2 = df2["lateness_bedtime"]


            elif y_col[0:9] == "programme":
                remove_list = [n for n in programmelist if n != y_col]
                df1 = df1.drop(remove_list, axis=1)
                df2 = df2.drop(remove_list, axis=1)
                y = df1["programme"]
                y2 = df2["programme"]
            else:
                y = df1[y_col]
                y2 = df2[y_col]




            df1 = df1.drop("chocolate", axis=1)
            df2 = df2.drop("chocolate", axis=1)
            # print(df1["chocolate_I have no idea what you are talking about"])
            # print(df2.columns)
            df1 = df1.drop("lateness_bedtime", axis=1)
            df2 = df2.drop("lateness_bedtime", axis=1)
            df1 = df1.drop("programme", axis=1)
            df2 = df2.drop("programme", axis=1)


            # try:
                # X, y = feature_selection(df, y_col)
            # X, y = feature_selection(df, y_col)
            try:
                X = df1.drop(y_col, axis=1)
                X2 = df2.drop(y_col, axis=1)
            except:
                X = df1
                X2 = df2

            # print(X2)
            X, y, X2 = feature_selection(X, X2, y, y2)
            # print(X2, y2)
            model = classify_KNN(X, y)
            accuracy = predictions(model, X2, y2)
            print("     acc:", accuracy, "features:", X2.columns)
            col_data.append(accuracy/100)

        data.append(col_data)
    plt.boxplot(data, labels=labels)
    plt.xticks(fontsize=8, rotation=30,ha='right', rotation_mode="anchor")

    # plt.xticks()
    # plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")
    plt.ylabel("Accuracy")

    plt.show()

                # if y_col == "chocolate":
                #     print("var:", col, "acc:", accuracy)

            # except:
            #     print("var:", col)

            # print("train: ", train)
            # print("test: ", test)
            # print(type(train))

    # for col in df.columns:

        # except:
        #     print(col)



    # X, y = decide_xy(df, ["informationretrieval", "statistics", "databases", "lateness_bedtime", "social"], "machinelearning")
