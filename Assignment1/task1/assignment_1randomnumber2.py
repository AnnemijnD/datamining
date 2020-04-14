import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('data/ODI-2020_cleaned.xlsx')
from datetime import datetime
import dateparser
from dateutil import parser

# print(df.head())

column_dict = {'What programme are you in?': 'programme',
            'Have you taken a course on machine learning?': 'machinelearning',
            'Have you taken a course on information retrieval?': 'informationretrieval',
            'Have you taken a course on statistics?': 'statistics',
            'Have you taken a course on databases?': 'databases',
            'What is your gender?': 'gender',
            'Chocolate makes you.....': 'chocolate',
            'When is your birthday (date)?': 'birthday',
            'Number of neighbors sitting around you?': 'neighbors',
            'Did you stand up?': 'stand',
            'What is your stress level (0-100)?': 'stresslevel',
            'You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? ': 'money',
            'Give a random number': 'randomnumber',
            'Time you went to be Yesterday': 'bedtime',
            'What makes a good day for you (1)?': 'goodday1',
            'What makes a good day for you (2)?': 'goodday2'}

df_new = df.rename(columns=column_dict)

# plt.boxplot([[1,2,3,3], [3,3,5,8]])
# plt.show()
counter = 0
numbers = []
hist = []
for index, row in df_new.iterrows():
    nr = row["randomnumber"]
    try:
        nr = float(nr)
        if nr >= 0 and nr <= 10:
            numbers.append(nr)
        else:
            counter += 1
            print(nr, "else")
            # hist.append(row["gender"])
            hist.append(row["informationretrieval"])
    except:
        counter += 1
        print(nr, "except")
        hist.append(row["informationretrieval"])

plt.title(f"Histogram uncorrect random number (n={len(hist)})")
plt.xlabel("Gender")
plt.ylabel("Frequency")
# print(counter)
# print(len())
plt.hist(hist)
plt.show()
