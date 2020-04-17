import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
import preprocessing


df_train, df_test = preprocessing.run_both()

print(df_train.columns)


# todo: values selecteren!
df_train = df_train.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Fare', 'Embarked'])
df_test = df_test.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Fare', 'Embarked'])
df_test_copy = df_test.drop(columns=['PassengerId']).copy()


df_train_features = df_train.drop(columns=['Survived'])
df_train_survived = df_train['Survived']

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(df_train_features, df_train_survived)
# test_pred = random_forest.predict(df_test)
score = random_forest.score(df_train_features, df_train_survived)
# acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
# acc_random_forest
print(score)

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
