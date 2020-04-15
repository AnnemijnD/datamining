import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
import preprocessing


df_train, df_test = preprocessing.run_both()

print(df_train.head())


def CV(df):

    kfold = KFold(10, True)

    # train and test are arrays with the row indices from the train_overall set
    for train, test in kfold.split(train_overall):

        # HIER HET MODEL
        # SLA OP HOE GOED DE PREDICTIONS WAREN

        break

    return kfold



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
