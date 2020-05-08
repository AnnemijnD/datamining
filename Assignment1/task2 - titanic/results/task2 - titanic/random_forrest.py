import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
import preprocessing as prep
from sklearn.model_selection import GridSearchCV
from neural_network import prepare_data_for_model
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


# load preprocessed data
df_train, df_test = prep.run_both()
df_test, X_train, y_train, X_test = prepare_data_for_model()

# remove passenger ID variable but save IDs for test set
df_train = df_train.drop(columns=['PassengerId'])
pass_id_test = df_test['PassengerId']
df_test = df_test.drop(columns=['PassengerId'])


def prediction(X_train, y_train, pass_id_test, X_test):
    """
    Make a prediction for the test set survival.
    """
    random_forest = RandomForestClassifier(n_estimators=50, min_samples_split=6, min_samples_leaf=2, max_depth=10)

    training = random_forest.fit(X_train, y_train)
    score = random_forest.score(X_train, y_train)
    print("Random forest score: ", score)

    prediction_test_set = random_forest.predict(X_test).round(0).astype(int)
    predictions = pd.DataFrame({'PassengerId': pass_id_test, 'Survived': prediction_test_set})
    predictions.to_csv('solutions/prediction_random_forrest_prep_min.csv', index=False)


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


def round_survival():
    """
    Round the survival results to integers for valid submission.
    """
    result = pd.read_csv("solutions/prediction_random_forrest_prep.csv")
    result["Survived"] = result["Survived"].round(0).astype(int)
    result.to_csv('solutions/prediction_random_forrest_prep.csv', index=False)


def model_testing(X_train, y_train):

    runs = 100

    acc1 = []
    acc2 = []
    acc3 = []

    for run in range(runs):
        model1 = RandomForestClassifier(n_estimators=50, min_samples_split=5, min_samples_leaf=2, max_depth=10)
        model2 = RandomForestClassifier(n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_depth=10)
        model3 = RandomForestClassifier(n_estimators=150, min_samples_split=5, min_samples_leaf=2, max_depth=10)

        training1 = model1.fit(X_train, y_train)
        training2 = model2.fit(X_train, y_train)
        training3 = model3.fit(X_train, y_train)

        acc1.append(model1.score(X_train, y_train))
        acc2.append(model2.score(X_train, y_train))
        acc3.append(model3.score(X_train, y_train))

    # save average accuracy of runs
    # val_accs.append(round(np.mean(acc_avg)*100, 2))
    # print("accuracy: " + str(np.mean(acc_avg)))

    # plot line for each activation method
    plt.boxplot([acc1,acc2,acc3])

    # plotting
    plt.title("Accuracy of random forest", fontsize=22)
    ax = plt.gca()
    ax.set_xticklabels(['50', '100', '150'], fontsize=18)
    plt.xlabel("# estimators")
    plt.ylabel("Accuracy score", fontsize=20)

    plt.subplots_adjust(bottom=.15, left=.15)
    # plt.savefig("-" + str(runs) + "runs.png")
    plt.show()

# param_tuning(X_train, y_train)
# prediction(X_train, y_train, pass_id_test, X_test)
model_testing(X_train, y_train)
# round_survival()

# # DIT HEEFT 63% GOED VOORSPELD, MET SIBSP ERBIJ 64 --> ALLEBEI ERG MATIG
# y = train_data["Survived"]
#
# features = ["Pclass", "Sex", "Parch", "SibSp"]
# X = pd.get_dummies(train_data[features])
# X_test = pd.get_dummies(test_data[features])
#
# print(X.columns)
#
# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(X, y)
#
# correct = 0
# incorrect = 0
# for index, row in X.iterrows():
#     pclass = row['Pclass']
#     sex_f = row['Sex_1']
#     sex_m = row['Sex_0']
#     parch = row['Parch']
#     sibsp = row['SibSp']
#     prediction = neigh.predict([[pclass, sex_f, sex_m, parch, sibsp]])
#
#     survived = train_data['Survived'][index]
#
#     if prediction == survived:
#         correct +=1
#     else:
#         incorrect += 1
#
# print(correct*100/(incorrect+correct))


# # TODO Dit is leuk bij exploratie erbij misschien!!!!
# alone = df_train.loc[df_train.alone == 0]["Survived"]
# not_alone = df_train.loc[df_train.alone == 1]["Survived"]
# rate_alone = sum(alone)/len(alone) # halve survived
# rate_not_alone = sum(not_alone)/len(not_alone) # 30% survived
# # print(rate_alone, rate_not_alone)
