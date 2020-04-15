import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# load data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

print(train_data.head())
print(test_data.head())

# percentage of men and women who survived
print(train_data[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean())
sns.countplot(x="Sex", data=train_data, hue="Survived")
plt.title("Survival per gender")
plt.show()

# survival per ticket type
sns.countplot(x="Pclass", data=train_data, hue="Survived")
plt.title("Survival per ticket type")
plt.show()

sns.countplot(x="Embarked", data=train_data, hue="Survived")
plt.title("Survival if embarked")
plt.show()

# check for missing values: Age, Cabin, Embarked
for col in train_data.columns.values:
    if train_data[col].isnull().any():
        print(f"Missing values in {col}")

# concatenated train and test data to find all relevant titles
data = pd.concat([train_data, test_data], ignore_index=True)

# obtain titles from passenger"s names
def add_titles(data):
    titles = []
    for row in data["Name"]:
        new=re.split(r"[,.]+", row)
        titles.append(new[1].strip())

    print(set(titles))

    # make new column with title category
    titles_cat = []
    for title in titles:
        if title in ["Don", "Sir", "Jonkheer"]:
            titles_cat.append("Noble male")
        elif title in ["the Countess", "Lady", "Dona"]:
            titles_cat.append("Noble female")
        elif title in ["Ms", "Miss", "Mlle"]:
            titles_cat.append("Miss")
        elif title in ["Mrs", "Mme"]:
            titles_cat.append("Mrs")
        elif title in ["Capt", "Col", "Dr", "Major", "Rev", "Master"]:
            titles_cat.append("Other")
        elif title == "Mr":
            titles_cat.append(title)

    data["Title"] = titles_cat
    return data

# add titles for train data
train_data = add_titles(train_data)
# print(train_data)

# survival per title
pd.factorize(train_data["Survived"])
print(train_data[["Title", "Survived"]].groupby(["Title"]).mean())
sns.countplot(x="Title", data=train_data, hue="Survived")
plt.title("Survival per title")
plt.show()

plt.figure(figsize=(10,3))
ax = sns.barplot(x="Title", y="Survived", data=train_data)
plt.show()

print("xxxxxxxxxxxxxxxxx",data.loc[data['Fare'].isnull()])
