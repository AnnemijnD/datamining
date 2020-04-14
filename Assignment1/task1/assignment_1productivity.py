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
words = ["productive",
"study",
"work",
"working",
"getting done",
"productivity",
"school",
"research",
"papers",
"assignment",
"coding",
"goals",
"achieve",
"competing",
"progress",
"accomplish"]
stresslevelsprod = []
stresslevelsothers = []
notwords = ["workout", "work out", "working out", "work-out"]

excluded = 0
correct = []
notcorrect = []
for index, row in df_new.iterrows():

    good1 = str(row["goodday1"]).lower()
    good2 = str(row["goodday2"]).lower()
    try:
        float(row["stresslevel"])
        if float(row["stresslevel"]) < 0 or float(row["stresslevel"]) > 100:
            continue
            excluded +=1

    except:
        excluded +=1
        continue


    if any(word in good1 for word in words):
        if not any(word in good1 for word in notwords):
            stresslevelsprod.append(float(row["stresslevel"]))
            correct.append(good1)
            continue
        # else:
        #     # print(good1)


    if any(word in good2 for word in words):
        if not any(word in good2 for word in notwords):
            stresslevelsprod.append(float(row["stresslevel"]))
            correct.append(good2)
        else:
            stresslevelsothers.append(float(row["stresslevel"]))
            notcorrect.append(good1)
            notcorrect.append(good2)
            # print(good2)
    else:
        stresslevelsothers.append(float(row["stresslevel"]))
        notcorrect.append(good1)
        notcorrect.append(good2)

print(notcorrect)
# print(len(stresslevelsprod), len(correct))
# print(len(stresslevelsothers), len(notcorrect))
plt.title(f"Stresslevels of people who are happy when productive, n={len(stresslevelsprod) + len(stresslevelsothers)}")
plt.ylabel("Stresslevel (0-100)")
plt.xlabel("People grouped by whether they have a good day when productive")
plt.boxplot([stresslevelsprod, stresslevelsothers])
plt.xticks([1,2],[f"Good day when productive, n={len(stresslevelsprod)}", f"Others, n={len(stresslevelsothers)}"])

plt.show()
