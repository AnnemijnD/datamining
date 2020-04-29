from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import clean_all_Data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn import svm
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle



def CV(df):
    train_overall, test_overall = train_test_split(df)

    kfold = KFold(10, True)

    # train and test are arrays with the row indices from the train_overall set
    for train, test in kfold.split(train_overall):


        break
def classify_KNN(X, y):

    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X, y)
    return neigh

def classify_svm(X, y):
    model = svm.SVC()
    model.fit(X, y)
    return model

def feature_selection(X, y, N):


    gain_vec = mutual_info_classif(X, y, discrete_features=True)

    delete_ind = gain_vec.argsort()[::-1][N:]
    delete = []
    for i in range(len(delete_ind)):
        delete.append(X.columns[i])

    # deletes the features that can be deleted
    X_fil = X.drop(delete, 1)

    # X_fil2 = X2.drop(delete, 1)


    return X_fil

def predictions(model, X, y):

    correct = 0
    incorrect = 0
    for index, row in X.iterrows():
        prediction = model.predict([row])

        answer = y.loc[index]
        if prediction == answer:
            correct +=1
        else:
            incorrect += 1

    accuracy = correct*100/(incorrect+correct)
    return accuracy

if __name__ == "__main__":


    df = clean_all_Data.run_all(False)
    df = df.dropna(subset=["gender"])
    df = df.reset_index(drop=True)
    df = df.drop(["programme", "chocolate"], axis=1)

    # df_check = df.dropna()
    # df_check = df_check.reset_index(drop=True)
    # y = df_check["gender"]
    # df_check = df_check.drop(['gender'], axis=1)
    # Ns = [2,3,4,5,6]
    boxplotdata = []
    labels = ["K2", "S2", "K3", "S3", "K4", "S4", "K5", "S5", "K6", "S6"]
    # for N in Ns:
    #     df_features = feature_selection(df_check, y, N)
    gender_features = ['chocolate_slim', 'chocolate_nan', 'lateness_bedtime', 'social', 'productive']
    #
    # gender_features = df_features.columns
    # print(gender_features)
    df = df.dropna(subset=gender_features)
    df = df.reset_index(drop=True)
    train_overall, test_overall = train_test_split(df)

    kfold = KFold(10, True)
    KNNacc = []
    SVMacc =[]
    iterations = 100
    print(df.shape)

    for iteration in range(iterations):
        print(iteration)
        # train and test are arrays with the row indices from the train_overall set
        for train, test in kfold.split(train_overall):
            df_test = df.drop(test.tolist(), axis=0) # groot
            df_train = df.drop(train.tolist(), axis=0) # klein
            df_test = df_test.reset_index() # groot
            df_train = df_train.reset_index() # klein

            #groot
            # print(df_test)

            #klein
            # print(df_train)

            x_test = df_test[gender_features]
            x_train = df_train[gender_features]
            y_test = df_test["gender"]
            y_train = df_train["gender"]

            model_KNN = classify_KNN(x_test, y_test)
            model_SVM = classify_svm(x_test, y_test)
            acc_knn = predictions(model_KNN, x_train, y_train)
            acc_svm = predictions(model_SVM, x_train, y_train)
            KNNacc.append(acc_knn)
            SVMacc.append(acc_svm)

    # plt.boxplot([KNNacc, SVMacc], labels=["KNN", "SVM"])
    # boxplotdata.append(KNNacc)
    # boxplotdata.append(SVMacc)
    plotdata = [KNNacc, SVMacc]
    with open('KNNSVMdata.pkl', 'wb') as f:
        pickle.dump(plotdata, f)
    # with open('KNNSVMdata.pkl', 'rb') as f:
    #     mynewlist = pickle.load(f)
    # print(mynewlist)
    plt.boxplot(plotdata, labels=["KNN", "SVM"])
    plt.show()
