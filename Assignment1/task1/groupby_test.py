import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re


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
# print(df.groupby([dict["What is your gender?"], dict["Number of neighbors sitting around you?"]]).count()[0])

# denken vrouwen dat chocolade hen dunner maakt???
gender_chocolate = df.groupby([dict["What is your gender?"], dict["Chocolate makes you....."]]).count()[0]

# print alleen vrouwen
# print(gender_chocolate["female"])

# alleen values
# print(gender_chocolate["female"].values)

# plotjes, nog niet echt nuttig
# plt.hist(gender_chocolate["female"])
# plt.show()


# stress level per study
# print(df.groupby([df[0], df[11]]).count())


# extract only relevant numbers
digits = []
non_digits = []
df2 = df
idxs = []

for nr, i in enumerate(df[10]):
    if (str(i).isdigit() or type(i) == int or type(i) == float) and i >= 0:
        digits.append(float(i))

    else:
        non_digits.append(i)
        idxs.append(nr)


df3 = df2.drop(index=idxs)

# print(df3.groupby([df3[10], df3[5]]).count())
gender = df3.groupby(df3[5])
print(gender.groups)

# make dictionary of groups and values per gender
gender_dict = {}
for gen, group in gender:
    gender_dict[gen] = group

stress_fem = list(gender_dict["female"][10].values)
stress_male = list(gender_dict["male"][10].values)
stress_unkn = list(gender_dict["unknown"][10].values)



# plt.hist(df3[10].values, bins=10)
# plt.hist(stress_fem)
# plt.boxplot(list(stress_fem))

# boxplot stress level per gender
plt.boxplot([stress_fem, stress_male, stress_unkn])
plt.xticks([1, 2, 3], ['Female', 'Male', 'Unknown'])
plt.title("Stress levels of students per gender")
plt.ylabel("Stress level")
plt.show()


# extract years from birth data
years = []
indxs = []

# iterate over data
for i, row in enumerate(df3[7]):
    remove = True

    # split date, append to list if year is given (delimeter ' ' (space) does not work)
    for mdy in re.split(r'[/.-]+', str(row)):
        if len(mdy) == 4 and int(mdy[0]) == 1 and int(mdy[1]) == 9:
                years.append(int(mdy))
                remove = False

    # if no year was found, add to to-remove indexes
    if remove:
        indxs.append(i)

# drop to-remove rows
df4 = df3.drop(index=indxs)
df4["years"] = years

# round years to nearest 5
round_y = []
for year in years:
    round_y.append(round(year / 5) * 5)

df4["round_y"] = round_y

# plot ages of students
plt.hist(years)
plt.title("Age of students")
plt.show()

# plot stress level per age group
plt.scatter(list(df4["round_y"].values), list(df4[10].values))
# sns.boxplot(df4["round_y"], df4[10])
plt.title("Stress level per age group of students")
plt.ylabel("Stress level")
plt.xlabel("Year")
plt.show()
