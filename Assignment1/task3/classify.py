import processdata
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn import svm
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from statistics import mean
from sklearn import preprocessing


N_features = 48

def get_data():

    df = processdata.run_all()
    print(df.head)
    df = df.drop("text", axis=1)

    return df

def classify_svm(X, y):

    model = svm.SVC()
    model.fit(X, y)
    return model

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

def make_conf(model, X, y):
    y_true = []
    y_pred = []
    for index, row in X.iterrows():
        prediction = model.predict([row])
        answer = y.loc[index]
        y_true.append(answer)
        y_pred.append(prediction.tolist()[0])
    lb = preprocessing.LabelBinarizer()
    # print(y_pred)
    # print("hoi", lb.fit([1, 2, 6, 4, 2]))
    # y_hoi = lb.fit(y_pred)
    # y_joe = lb.fit(y_true)
    # print(y_hoi, y_joe)
    print("hihihhii")
    precision = precision_score(y_true, y_pred, pos_label='spam')
    recall = recall_score(y_true, y_pred,pos_label="spam")

    print(precision, recall)
    confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None)
    titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(model, X, y,
                                     display_labels=["spam", "ham"],
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()

def feature_selection(X,y):

    gain_vec = mutual_info_classif(X, y, discrete_features=True)

    delete_ind = gain_vec.argsort()[::-1][N_features:]
    delete = []

    for i in range(len(delete_ind)):
        delete.append(X.columns[i])

    # deletes the features that can be deleted
    X_fil = X.drop(delete, 1)


    return X_fil


def run():
    df = get_data()
    ycol = "label"
    y = df["label"]
    X = df.drop("label", axis=1)
    print(X.shape)
    # features = feature_selection(X, y).columns
    features = X.columns
    model_SVM = classify_svm(X, y)
    make_conf(model_SVM, X, y)
    # df = df.drop("label", axis=1)

    # model = classify_svm(X, y)
    SVMacc = []
    iterations = 10
    for iteration in range(iterations):
        train_overall, test_overall = train_test_split(df)

        kfold = KFold(5, True)
        print(iteration)

        # train and test are arrays with the row indices from the train_overall set
        for train, test in kfold.split(train_overall):
            df_test = df.drop(test.tolist(), axis=0) # groot
            df_train = df.drop(train.tolist(), axis=0) # klein
            df_test = df_test.reset_index(drop=True) # groot
            df_train = df_train.reset_index(drop=True) # klein


            x_test = df_test
            x_train = df_train
            y_test = df_test["label"]
            y_train = df_train["label"]
            x_test = x_test.drop("label", axis=1)
            x_train = x_train.drop("label", axis=1)


            x_test = x_test[features]
            x_train = x_train[features]
            model_SVM = classify_svm(x_test, y_test)
            acc_svm = predictions(model_SVM, x_train, y_train)
            SVMacc.append(acc_svm)

    # acc = predictions(model, X, y)
    print(len(SVMacc))
    print(mean(SVMacc))

    mean_svm = mean(SVMacc)/100
    n = len(SVMacc)
    z = 1.96

    element = z * np.sqrt(mean_svm * (1-mean_svm) / n)
    upper = mean_svm + element
    lower = mean_svm - element

    print("SVM: [", lower, ",", upper, "]")

    print(SVMacc)
    plt.boxplot(SVMacc)
    plt.show()


    # print("Acc", acc)

if __name__ == "__main__":
    run()
