from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import numpy as np
import pandas as pd
from preprocessing import prep_data


def sim(a, b):

    sim = np.sum((r_ap - r_amean) * (r_bp - r_bmean)) / (np.sqrt(np.sum(r_ap - r_amean)**2) * np.sqrt(np.sum(r_bp - r_bmean)**2))

    return sim


def pred(a, p):

    pred = r_amean + np.sum(sim(a, b) * (r_bp - r_bmean)) / np.sum(sim(a, b))


def param_tuning(X_train, y_train):
    """
    Test the hyperparameters to obtain optimal accuracy on the test set.
    """
    for i in tqdm(range(10)):

        time.sleep(3)

        parameters = {
        'n_estimators': [10, 30, 50, 70, 100, 500],
        'max_depth': [1, 5, 10, 15],
        "max_terminal_nodes": [25, 50],
        'min_samples_split': [5,8,10],
        'min_samples_leaf': [.1, .2, .5, 2, 5],
        }

        rf = RandomForestClassifier(max_samples=.2)

        # Using a grid search with a 5-fold cross validation to find the best model
        rf_clf = GridSearchCV(rf, parameters, scoring='accuracy', cv=5)
        rf_clf.fit(X_train, y_train)

        print('Random Forrest')
        print(rf_clf.best_params_)
        print(f'Accuracy: {round(rf_clf.best_score_*100, 2)}%')


if __name__ == "__main__":

    # load data
    df_train = pd.read_csv("data/training_short.csv")
    # df_train = pd.read_csv("data/training_set_VU_DM.csv")
    df_test = pd.read_csv("data/test_short.csv")

    data, df_test = prep_data(df_train, df_test)


    predictors = [c for c in data.columns if c not in ["booking_bool","click_bool","gross_bookings_usd","position"]]

    X = data[predictors]
    y = data.booking_bool.astype(int)

    clf = AdaBoostClassifier(n_estimators=100)
    training = clf.fit(X, y)
    score = clf.score(X, y)
    print(score, "score")
    scores = cross_val_score(clf, X, y, cv=5)
    print("mean score: ", scores.mean())
    print("ada scores:")
    print(scores)

    prediction_test_set = clf.predict(df_test)
    predictions = pd.DataFrame({'hotel_id': df_test.prop_id, 'search_id': df_test.srch_id, 'booking_prob': prediction_test_set})
    predictions.to_csv('wattt.csv', index=False)
    
    clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
    scores = cross_val_score(clf, X, y, cv=3)
    print("mean score: ", scores.mean())
    print("rf scores:")# scores = cross_validate(clf, data[predictors], data['booking_bool'], cv=3)
    print(scores)
