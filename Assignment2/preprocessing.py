import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


# load data
df_train = pd.read_csv("data/training_set_VU_DM.csv")
# df_test = pd.read_csv("data/test_set_VU_DM.csv")

print(df_train.head(10))
# print(df_test.head(10))

# datasets = [df_train, df_test]
# for data in datasets:
data = df_train.sample(n=1000)
# get latex table for summaries of categorical and numerical data
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
print('numeric columns: ' + str(numeric_columns))
print(round(data[numeric_columns].describe(),2)) # MEMORYERROR
print('categorical columns: ' + str(categorical_columns))
print(data[categorical_columns].describe())
# quit()
# drop categorical
data.drop(categorical_columns, axis=1, inplace=True)

print("MISSING VALS BEFORE")
for col in data.columns.values:
    if data[col].isnull().any():
        print(f"Missing values in {col}")

for var in numeric_columns:
    data[var].fillna(data.groupby("prop_starrating")[var].transform("mean"), inplace=True)

to_drop = []
print("MISSING VALS AFTER")
for col in data.columns.values:
    if data[col].isnull().any():
        to_drop.append(col)
        
data.drop(to_drop, axis=1, inplace=True)

# scale numeric
scaler = StandardScaler()

for var in numeric_columns:
    data[var] = data[var].astype("float64")
    data[var] = scaler.fit_transform(data[var].values.reshape(-1, 1))

# importance
target = data['position'].values
select_features = data.columns.values

selector = SelectKBest(f_classif, len(select_features))
selector.fit(data, target)
scores = -np.log10(selector.pvalues_)
indices = np.argsort(scores)[::-1]

print('Features importance:')
for i in range(len(scores)):
    print('%.2f %s' % (scores[indices[i]], select_features[indices[i]]))
