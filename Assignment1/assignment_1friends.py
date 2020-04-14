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
words = ["friend", "social", "family"]
friendsgroup = []
others = []
notwords = []

excluded = 0
correct = []
notcorrect = []

for index, row in df_new.iterrows():

    good1 = str(row["goodday1"]).lower()
    good2 = str(row["goodday2"]).lower()
    # try:
    #     float(row["stresslevel"])
    #     if float(row["stresslevel"]) < 0 or float(row["stresslevel"]) > 100:
    #         continue
    #         excluded +=1
    #
    # except:
    #     excluded +=1
    #     continue


    if any(word in good1 for word in words):
        if not any(word in good1 for word in notwords):
            friendsgroup.append(row["gender"])
            correct.append(good1)
            continue
        # else:
        #     # print(good1)


    if any(word in good2 for word in words):
        if not any(word in good2 for word in notwords):
            friendsgroup.append(row["gender"])
            correct.append(good2)
        else:
            others.append(row["gender"])
            notcorrect.append(good1)
            notcorrect.append(good2)
            # print(good2)
    else:
        others.append(row["gender"])
        notcorrect.append(good1)
        notcorrect.append(good2)

female = [0,0]
male = [0,0]
unknown = [0,0]
female_total = 0
male_total = 0
unknown_total = 0
for i in range(len(friendsgroup)):
    if friendsgroup[i] == "female":
        female[0] += 1
        female_total += 1
    elif friendsgroup[i] == "male":
        male[0] += 1
        male_total += 1
    else:
        unknown[0] += 1
        unknown_total += 1
for i in range(len(others)):
    if others[i] == "female":
        female[1] += 1
        female_total += 1
    elif others[i] == "male":
        male[1] += 1
        male_total += 1
    else:
        unknown[1] += 1
        unknown_total += 1

female[0] = female[0]/female_total
female[1] = female[1]/female_total
male[0] = male[0]/male_total
male[1] = male[1]/male_total
unknown[0] = unknown[0]/unknown_total
unknown[1] = unknown[1]/unknown_total

barWidth = 0.33

# Set position of bar on X axis
r1 = np.arange(len(female))
r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]

sociallist = [female[0], male[0]]
otherslist =[female[1], male[1]]
# Make the plot
# plt.bar(r1, female, width=barWidth, edgecolor='white', label=f'Female, n={female_total}')
# plt.bar(r2, male, width=barWidth, edgecolor='white', label=f'Male, n={male_total}')
plt.bar(r1, sociallist, width=barWidth, edgecolor='white', label=f'Social')
plt.bar(r2, otherslist, width=barWidth, edgecolor='white', label=f'Others')

# plt.bar(r3, unknown, width=barWidth, edgecolor='white', label='Unknown')

# Add xticks on the middle of the group bars
plt.xlabel('Group', fontweight='bold')
plt.xticks([0.165,1.165], ["Female", "Male"])
plt.ylabel("Fraction")
plt.title("People that have a good day when being social vs gender")
# plt.xticks([r + barWidth for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])

# Create legend & Show graphic
plt.legend()
plt.show()






# print(notcorrect)
# print(len(stresslevelsprod), len(correct))
# print(len(stresslevelsothers), len(notcorrect))

# plt.boxplot([friendsgroup, stresslevelsothers])
