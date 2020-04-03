import pandas as pd
# import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np

# import data
filename = 'data/ODI-2020.xlsx'
df = pd.read_excel(filename)

# change colnames
cols = df.columns.values
dict = {}
for i, name in enumerate(cols):
    dict[name] = i
df.rename(columns=dict, inplace=True)

# kijken welke vakken gevolgd
print(df.groupby([df[0],df[1]]).count())

# je kunt ook .mean(à gebruiken ipv .count maar dan hebben we numerieke waarden nodig)
# kijken welke vakken gevolgd per richting
# print(df.groupby([df[0],df[1],df[2]]).count())

# hoeveel mensen naast je per gender
print(df.groupby([dict["What is your gender?"], dict["Number of neighbors sitting around you?"]]).count()[0])

# denken vrouwen dat chocolade hen dunner maakt???
gender_chocolate = df.groupby([dict["What is your gender?"], dict["Chocolate makes you....."]]).count()[0]

# print alleen vrouwen
print(gender_chocolate["female"])

# alleen values
print(gender_chocolate["female"].values)

# plotjes, nog niet echt nuttig
plt.hist(gender_chocolate["female"])
plt.show()
