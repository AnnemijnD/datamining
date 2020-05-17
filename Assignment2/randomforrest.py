import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import xgboost as xgb # pip install first


def prediction(df_test, X_train, y_train, X_test):
    """
    Make a prediction for the test set survival.
    """
    model = RandomForestClassifier(n_estimators=50, min_samples_split=5, min_samples_leaf=5, max_depth=15)

    training = model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    print("Random forest score: ", score)

    df_test['category'] = model.predict(X_test)
    solution = df_test[['srch_id', 'prop_id', 'category']]
    date_time = time.strftime("%Y-%m-%d-%H-%M")
    solution = solution.sort_values(by='category', ascending=False)
    solution = solution.drop("category", axis=1)
    solution.to_csv('results/solutions/randomf_' + str(date_time) + ".csv", index=False)
    return score


def param_tuning(X_train, y_train):
    """
    Test the hyperparameters to obtain optimal accuracy on the test set.
    """
    # for i in tqdm(range(10)):

    parameters = {
        'n_estimators': [10, 30, 50],
        'max_depth': [5, 10, 15],
        # "max_terminal_nodes": [5, 10],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 5, 7],
    }

    rf = RandomForestClassifier(max_samples=.2)

    # Using a grid search with a 5-fold cross validation to find the best model
    grid = GridSearchCV(rf, parameters, scoring='accuracy', cv=5)
    result = grid.fit(X_train, y_train)

    means = result.cv_results_['mean_test_score']
    stds = result.cv_results_['std_test_score']
    params = result.cv_results_['params']

    print(means)
    print(stds)
    print(params)

    print('Random Forrest')
    print(grid.best_params_)
    print(f'Accuracy: {round(grid.best_score_*100, 2)}%')


def XGBoost(Xtrain, Ytrain, df_test, Xtest):
    X_train, X_test, y_train, y_test = train_test_split(Xtrain, Ytrain, test_size=0.33, random_state=7)
    # gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)

    """ training """
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    """ prediction """
    df_test['category'] = model.predict(Xtest)

    solution = df_test[['srch_id', 'prop_id', 'category']]

    date_time = time.strftime("%Y-%m-%d-%H-%M")
    solution = solution.sort_values(by='category', ascending=False)
    solution = solution.drop("category", axis=1)
    print(solution.head())
    solution.to_csv('results/solutions/xgboost_' + str(date_time) + ".csv", index=False)



if __name__ == "__main__":
    # df_train = pd.read_csv("data/train_selection.csv")
    # df_train = pd.read_csv("data/training_set_VU_DM.csv")
    df_test = pd.read_csv("data/test_prep_long.csv")
    df_train = pd.read_csv("data/train_prep_long.csv")
    # df_test = pd.read_csv("data/test_set_VU_DM.csv")

    predictors = [c for c in df_train.columns if c not in ["prop_id","srch_id","booking_bool",\
                                "click_bool","gross_bookings_usd","position", "category"]]
    X_train = df_train[predictors]

    # predicting columns of test set
    cols = [col for col in df_test.columns if col not in ['prop_id', 'srch_id']]
    X_test = df_test[cols]
    y_train = df_train.category.astype(int)

    """ functions """
    prediction(df_test, X_train, y_train, X_test)
    # param_tuning(X_train, y_train)
    # XGBoost(X_train, y_train, df_test, X_test)
