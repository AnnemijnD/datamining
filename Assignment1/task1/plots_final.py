import pandas as pd
import matplotlib.pyplot as plt
import clean_all_Data
import numpy as np
import scipy.stats as sts

def gender_social(df):
    male = [0,0]
    female = [0,0]

    all_male = 0
    all_female = 0

    for index, row in df.iterrows():

        if row["social"] == 1:
            if row["gender"] == 1:
                female[0] += 1
                all_female += 1
            else:
                male[0] += 1
                all_male+=1
        else:
            if row["gender"] == 1:
                female[1] += 1
                all_female += 1
            else:
                male[1] += 1
                all_male+=1

    female[0] = female[0]/all_female
    female[1] = female[1]/all_female
    male[0] = male[0]/all_male
    male[1] = male[1]/all_male

    print(female, male)
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
    plt.title("Good day when social and gender")
    # plt.xticks([r + barWidth for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])

    # Create legend & Show graphic
    plt.legend()
    plt.show()

# def statistics_bedtime(df):
#     yes = []
#     no = []
#     df = df.dropna(subset=["statistics", "lateness_bedtime"])
#     # df = df["lateness_bedtime"].dropna()
#     df = df.reset_index(drop=True)
#
#     for index, row in df.iterrows():
#
#
#
#         if row["statistics"] == 0:
#             print(row["lateness_bedtime"])
#             no.append(row["lateness_bedtime"])
#         elif row["statistics"] == 1:
#             print(row["lateness_bedtime"])
#             yes.append(row["lateness_bedtime"])
#
#     print(yes)
#     print(no)
#     plt.boxplot([yes, no], labels=["yes", "no"])
#     plt.show()

# def bedtime_productive(df):
#
#     yes = []
#     no = []
#
#     df = df.dropna(subset=["productive", "lateness_bedtime"])
#     # df = df["lateness_bedtime"].dropna()
#     df = df.reset_index(drop=True)
#
#     for index, row in df.iterrows():
#         if row["productive"] == 0:
#             # print(row["lateness_bedtime"])
#             no.append(row["lateness_bedtime"])
#         elif row["productive"] == 1:
#             # print(row["lateness_bedtime"])
#             yes.append(row["lateness_bedtime"])
#
#     # print(yes)
#     # print(no)
#     plt.boxplot([yes, no], labels=["yes", "no"])
#     plt.show()


# def neighbors_gender(df):
#
#     male = []
#     female = []
#
#     df = df.dropna(subset=["gender", "neighbors"])
#     # df = df["lateness_bedtime"].dropna()
#     df = df.reset_index(drop=True)
#
#     for index, row in df.iterrows():
#         if row["gender"] == 0:
#             # print(row["lateness_bedtime"])
#             male.append(row["neighbors"])
#         elif row["gender"] == 1:
#             # print(row["lateness_bedtime"])
#             if row["neighbors"] > 80:
#                 continue
#             female.append(row["neighbors"])
#
#     # print(yes)
#     # print(no)
#     plt.boxplot([male, female], labels=["male", "female"])
#     plt.show()

def bedtime_gender(df):

    male = []
    female = []

    df = df.dropna(subset=["gender", "lateness_bedtime"])
    # df = df["lateness_bedtime"].dropna()
    df = df.reset_index(drop=True)

    for index, row in df.iterrows():
        if row["gender"] == 0:
            # print(row["lateness_bedtime"])
            male.append(row["lateness_bedtime"])
        elif row["gender"] == 1:
            # print(row["lateness_bedtime"])

            female.append(row["lateness_bedtime"])

    # print(yes)
    # print(no)
    plt.boxplot([male, female], labels=["male", "female"])
    plt.show()

def bedtime_stresslevel(df):
    x = []
    y = []
    df = df.dropna(subset=["stress", "lateness_bedtime"])
    df = df.reset_index(drop=True)
    df = df[df["lateness_bedtime"].between(df["lateness_bedtime"].quantile(.15), df["lateness_bedtime"].quantile(.85))]
    for index, row in df.iterrows():
        # print(row["stress"], row["lateness_bedtime"])
        plt.scatter(row["stress"], row["lateness_bedtime"], color="blue")
        x.append(row["stress"])
        y.append(row["lateness_bedtime"])
    print(sts.pearsonr(x, y))


    plt.show()

def randomnumber(df):
    numbers = []
    counter = 0
    for index, row in df.iterrows():
        numbers.append(row["randomnumber"])
        if row["randomnumber"] >= 0 or row["randomnumber"] <= 10:
            counter +=1




    plt.title(f"Histogram random numbers (n={counter})")
    plt.xlabel("Random numbers")
    plt.ylabel("Frequency")

    plt.hist(numbers, histtype='bar', ec='black')
    plt.show()

def sannesplot(df):
    drop_list = []
    for index, row in df.iterrows():
        if type(row['money']) == str or type(row['stress']) == str or type(row['gender']) == str:
            drop_list.append(index)

    df = df.drop(drop_list)

    options = {}
    doubles = {}

    # select values that occur multiple times (doubles)
    for index, row in df.iterrows():

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

    # set gender color
    for index, row in df.iterrows():
        coordinate = (row['money'], row['stress'])
        if coordinate in doubles.keys():
            df.at[index, "gender"] = doubles[coordinate][0]

    # add size to df
    df['size'] = len(df) * [5]
    for index, row in df.iterrows():
        coordinate = (row['money'], row['stress'])
        if coordinate in doubles.keys():
            df.at[index, 'size'] = (5 + doubles[coordinate][1] * 2)

    df['money'] = pd.to_numeric(df['money'])
    df['stress'] = pd.to_numeric(df['stress'])

    ax = df.plot.scatter(x='money', y='stress', s=df['size'], c='gender', colormap='cool')
    ax.plot()
    ax.set_title("Money in relation to stresslevel for different gender", fontsize=14)
    ax.set_ylabel("Stresslevel", fontsize=12)
    ax.set_xlabel("Money", fontsize=12)
    plt.show()





df = clean_all_Data.run_all(True)
# bedtime_stresslevel(df)


# bedtime_gender(df)
# randomnumber(df)
# gender_social(df)
# sannesplot(df)
