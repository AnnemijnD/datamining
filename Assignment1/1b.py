from sklearn.model_selection import train_test_split, KFold
import pandas as pd

df = pd.read_excel('data/ODI-2020.xlsx')

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
