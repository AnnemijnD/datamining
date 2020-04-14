import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_excel('data/ODI-2020_cleaned.xlsx')

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
            'What is your stress level (0-100)?': 'stress',
            'You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? ': 'money',
            'Give a random number': 'randomnumber',
            'Time you went to be Yesterday': 'bedtime',
            'What makes a good day for you (1)?': 'goodday1',
            'What makes a good day for you (2)?': 'goodday2'}

df_new = df.rename(columns=column_dict)

drop_list = []
for index, row in df_new.iterrows():
    if type(row['stress']) == str:
        drop_list.append(index)
    elif row['stress'] < 0 or row['stress'] > 100:
        drop_list.append(index)
    elif type(row['bedtime']) == float:
        drop_list.append(index)

df_stress_sleep = df_new.drop(drop_list)

print(df_stress_sleep)

stress_list = []
sleep_list = []
for index, row in df_stress_sleep.iterrows():
    stress_list.append(row['stress'])
    sleep_list.append(row['bedtime'])

plt.scatter(sleep_list, stress_list)
plt.xlabel("Bedtime")
plt.ylabel("Stresslevel")
plt.show()
