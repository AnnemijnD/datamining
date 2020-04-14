"""
DEZE WERKT NU EFFIES NIET MEER OMDAT DE DATABASE ANDERS IS NU
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import clean_all_Data

df = clean_all_Data.run_all()

#
# # Make df where all the money values are valid (unvalid values removed --> nog niet gekeken naar 'euros' verwijderen etc.)
# drop_list = []
# for index, row in df_new.iterrows():
#     if type(row['money']) == str:
#         drop_list.append(index)
#     elif row['money'] < 0 or row['money'] > 100:
#         drop_list.append(index)
#     else:
#         # print(type(row['money']))
#         row['money'] = float(row['money'])
#
# df_money = df_new.drop(drop_list)
#
# print(len(drop_list))

# money_f, money_m = [], []
# for index, row in df_money.iterrows():
#     if row['gender'] == 'female':
#         money_f.append(row['money'])
#     elif row['gender'] == 'male':
#         money_m.append(row['money'])



# # Boxplot vrouw vs man, met op yas hoeveel geld
# plt.boxplot([money_f, money_m])
# plt.show()



# len_diff = len(money_m) - len(money_f)
# for i in range(len_diff):
#     money_f.append(None)
#
# dict_mg = {'female': money_f, 'male': money_m}
# df_mg = pd.DataFrame(dict_mg, columns = ['female', 'male'])
# print(df_mg)

# plt.hist(money_f)
# plt.show()
# plt.hist(money_m)
# plt.show()

# money_gender = df_money.groupby(['gender', 'money']).count()['programme']

# Mean van vrouwen is hoger!!! TODO: fixen om histogrammen te maken
# female_money = money_gender['female']
# print(female_money.mean())
# print(money_gender['male'].mean())
# print(type(female_money))
#
# print(female_money.plot.hist())
# plt.show(female_money.plot.hist())
#
# plt.hist(female_money)
# plt.invert_yaxis()
# plt.show()

# TODO: evt. kan dit het probleem oplossen, eerst even kieken naar money en stress
# money_gender = df_money.groupby(['money', 'gender']).count()['programme']
#
# # print(money_gender)
# # print(type(money_gender))

print(df)

# 3 rijen worden hierbij verwijderd. Dit is dus wel echt alleen de combi met correcte money
drop_list_stress = []
stress_list = []
money_list = []
both_list = []
gender_list = []
for index, row in df_money.iterrows():
    if type(row['stress']) == str:
        drop_list_stress.append(index)
    elif row['stress'] < 0 or row['stress'] > 100:
        drop_list_stress.append(index)
    else:
        stress_list.append(row['stress'])
        money_list.append(row['money'])
        both_list.append([row['stress'], row['money']])
        gender_list.append(row['gender'])

# set gender to integer, 0 for male and 1 for female
df_money_stress = df_money.drop(drop_list_stress)
drop_gender = []
for index, row in df_money_stress.iterrows():
    if row['gender'] == 'female':
        row['gender'] = 1
    elif row['gender'] == 'male':
        row['gender'] = 0
    else:
        drop_gender.append(index)

# drop rows for which the gender is unknown
df_money_stress = df_money_stress.drop(drop_gender)

stress_f, stress_m = [], []
options = {}
doubles = {}

for index, row in df_money_stress.iterrows():
    if row['gender'] == 'female':
        stress_f.append(row['stress'])
    elif row['gender'] == 'male':
        stress_m.append(row['stress'])

    # make dictionary with double values in the scatterplot
    coordinate = (row['money'], row['stress'])
    if coordinate in options.keys():
        if coordinate not in doubles.keys():
            doubles[coordinate] = [row['gender'], options[coordinate]]
        else:
            doubles[coordinate].append(row['gender'])

    # keep track of all coordinates
    options[coordinate] = row['gender']


# change doubles into average and amount of people
for i in doubles.keys():
    average = sum(doubles[i])/len(doubles[i])
    doubles[i] = [average, len(doubles[i])]


for index, row in df_money_stress.iterrows():
    coordinate = (row['money'], row['stress'])
    if coordinate in doubles.keys():
        row['gender'] = doubles[coordinate][0]

df_money_stress['size'] = len(df_money_stress) * [5]

for index, row in df_money_stress.iterrows():
    coordinate = (row['money'], row['stress'])
    if coordinate in doubles.keys():
        df_money_stress.at[index, 'size'] = (5 + doubles[coordinate][1] * 2)

df_money_stress['money'] = pd.to_numeric(df_money_stress['money'])
df_money_stress['stress'] = pd.to_numeric(df_money_stress['stress'])


ax = df_money_stress.plot.scatter(x='money', y='stress', s=df_money_stress['size'], c='gender', colormap='cool')
ax.plot()
ax.set_title("Money in relation to stresslevel for different gender", fontsize=14)
ax.set_ylabel("Stresslevel", fontsize=12)
ax.set_xlabel("Money", fontsize=12)
plt.show()



# # Vrouwen hebben iets meer stress dan mannen
# print(sum(stress_f)/len(stress_f))
# print(sum(stress_m)/len(stress_m))
# plt.boxplot([stress_f, stress_m])
# plt.xticks([1, 2], ["Female", "Male"])
# plt.ylabel("Stresslevel")
# plt.show()

# money_stress = df_money_stress.groupby(['stress', 'money', 'gender']).count()['programme']

# print(money_stress)
# print(type(money_stress))


# size_dict = {}
# for combination in set(both_list):
#     both_list.count(combination)


# plt.scatter(stress_list, money_list)
# plt.xlabel("Stress level")
# plt.ylabel("Money")
# plt.show()
