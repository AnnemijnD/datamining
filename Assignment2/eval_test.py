import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def eval():

    true_relevance = np.asarray([[10, 0, 0, 1, 5]])
    scores = np.asarray([[1, 0, 0, 0, 1]])
    ndcg_score(true_relevance, scores,k=5)

    score = 0
    return score



def prediction(df_test, X_train, y_train, X_test):
    """
    Make a prediction for the test set survival.
    """
    model = RandomForestClassifier(n_estimators=30, min_samples_split=10, min_samples_leaf=2, max_depth=5)

    training = model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    print("Random forest score: ", score)

    # df_test['category'] = model.predict(X_test)
    # solution = df_test[['srch_id', 'prop_id', 'category']]
    # predictions.to_csv('solutions/prediction_random_forrest_prep_min.csv', index=False)

    return score


if __name__ == "__main__":
    # df_train = pd.read_csv("data/train_selection.csv")
    # df_train = pd.read_csv("data/training_set_VU_DM.csv")
    # df_test = pd.read_csv("data/test_prep_long.csv")
    # df_test = pd.read_csv("data/test_set_VU_DM.csv")

    df_train = pd.read_csv("data/train_prep_long.csv")
    print(df_train.columns)
    exit()
    predictors = [c for c in df_train.columns if c not in ["prop_id","srch_id","booking_bool",\
                                "click_bool","gross_bookings_usd","position", "category"]]
    X_train = df_train[predictors]

    # predicting columns of test set
    # cols = [col for col in df_test.columns if col not in ['prop_id', 'srch_id']]
    # X_test = df_test[cols]
    y_train = df_train.category.astype(int)
    print("hoi")
    """ functions """
    # prediction(df_test, X_train, y_train, X_test)
    # param_tuning(X_train, y_train)
