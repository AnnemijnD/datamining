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
#
# plt.boxplot([[1,2,3,3], [3,3,5,8]])
# plt.show()

dict = {"slim":[], "fat":[], "I have no idea what you are talking about":[], "neither":[], "unknown":[]}
times = []
times_str = []
times_h = []
times_min = []
times_min_rel = []
# print(df_new)
# valid_entries = 0
values = []
for index, row in df_new.iterrows():
    answer = row["chocolate"]
    bedtime = row["bedtime"]
    # print(bedtime)
    times.append(bedtime)
    try:
        bedtime1 = bedtime.strftime("%H:%M:%S")[0:-3]
        bedtime1 = bedtime1.replace(":",".")
        # print(bedtime)
        # print(bedtime%19)
        times_str.append(float(bedtime1))

        # print(bedtime[::-1:-3])
        h = int(bedtime1[:len(bedtime1)-3])
        times_h.append(h)

        bedtime2= bedtime.strftime("%M")
        bedtime2 = bedtime2.replace(":",".")
        # print(bedtime2)
        new_min = float(bedtime2)/60
        times_min.append(bedtime2)
        times_min_rel.append(new_min)
        values.append(answer)

            # print("hi")
    except:
        pass

# print(max(times_str))
# print(times_h)
relativetimes = []
# print(times_h)
# print(times_min_rel)
dict = {19: 0, 20:1, 21:2, 22:3, 23:4, 0:5, 1:6, 2:7, 3:8, 4:9, 5:10, 6:11, 7:12}

for i in range(len(times_h)):
    new_timestamp = float(str(dict[times_h[i]]) + str(times_min_rel[i])[1:len(times_min_rel) -1])
    print(new_timestamp)
    relativetimes.append(new_timestamp)



dict2 = {"slim":[], "fat":[], "I have no idea what you are talking about":[], "neither":[], "unknown":[]}
print(values)
for i in range(len(relativetimes)):
    dict2[values[i]].append(relativetimes[i])

labels = ["slim","fat","I have no idea","neither","unknown"]
yticksl = ["19:00", "20:00", "21:00", "22:00", "23:00", "00:00","01:00", "02:00",
          "03:00", "04:00", "05:00", "06:00", "07:00"]
# dfbox = pd.DataFrame.from_dict(dict2)
# print(dfbox)
# dfbox = df.boxplot(column=list(dict2.keys()))
plt.boxplot(list(dict2.values()))
plt.xticks([1, 2, 3,4,5], labels)
plt.yticks([0,1,2,3,4,5,6,7,8,9,10,11,12], yticksl)
plt.show()



    # try:
    #     print(datetime.parse(bedtime))
    #
    # except:
    #     print("nope")
    # #
    # # try:
    #
    #     bedtime = float(row["bedtime"])
    #     dict[answer].append(bedtime)
    #     print(bedtime)
    # except:
    #     print("kon nie", bedtime)

        # continue
# dfbox = pd.DataFrame.from_dict(dict)
# print(dfbox)
# dfbox = df.boxplot(column=list(dict.keys()))
# plt.boxplot(list(dict.values()))

# print(list(dict.values()))
# plt.show()
