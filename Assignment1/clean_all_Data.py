import pandas as pd
import numpy as np
import dateparser
from datetime import datetime
from datetime import date as date_now

def new_column_names(df):
    """
    Renames the columns
    """
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
    return df_new

def birthyears(df):
    """
    Returns a dataframe with birthyears instead of birthdays. If birthyear was undefined,
    "NaN" is used.
    """
    # print(df['You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? '])
    # print(df.columns)

    counter = 0
    datelist = []
    year_list = []
    for index, row in df.iterrows():
        date = str(row['birthday'])
        if dateparser.parse(date) is not None:
            new_date = dateparser.parse(date)
            datelist.append(dateparser.parse(date))
            # print("year", new_date.year)
            year = new_date.year
            if year > 2001 or year < 1975:

                df.at[index, 'birthday'] = "NaN"
            else:
                df.at[index, 'birthday'] = year

        else:
            df.at[index, 'birthday'] = "NaN"
            counter +=1

    # print(counter)
    return df

def programme(df):
    # capital check
    CLS = ["cls", "computational science"]
    AI = ["ai", "artificial intelligence"]
    BA = ["ba", "business analytics"]
    CS = ["cs", "computer science"]
    BF = ["bioinformatics"]
    econometrics = ["econometrics"] # '& operations research' wordt niet als losse master gezien nu
    QRM = ["qrm", "quantitative risk management"]

    for index, row in df.iterrows():
        programme = row["programme"].lower()
        if any(word in programme for word in CLS):
            df.at[index, "programme"] = "CLS"
        elif any(word in programme for word in AI):
            df.at[index, "programme"] = "AI"
        elif any(word in programme for word in BA):
            df.at[index, "programme"] = "BA"
        elif any(word in programme for word in CS):
            df.at[index, "programme"] = "CS"
        elif any(word in programme for word in BF):
            df.at[index, "programme"] = "BF"
        elif any(word in programme for word in econometrics):
            df.at[index, "programme"] = "econometrics"
        elif any(word in programme for word in QRM):
            df.at[index, "programme"] = "QRM"
        else:
            df.at[index, "programme"] = "other"

    return df


def MC(df):
    answer_dict0 = {"no": 0, "yes": 1, "unknown": "NaN"}
    answer_dict1 = {0: 0, 1: 1, "unknown": "NaN"}
    answer_dict2 = {"mu": 0, "sigma": 1, "unknown": "NaN"}
    answer_dict3 = {"nee": 0, "ja": 1, "unknown": "NaN"}
    answer_dict4 = {"male": 0, "female": 1, "unknown": "NaN"}
    answer_dict5 = {"fat":0, "slim":1, "I have no idea what you are talking about":2,
                    "neither":3, "unknown":4}

    for index, row in df.iterrows():

        # machine learning
        df.at[index, "machinelearning"] = answer_dict0[row["machinelearning"]]

        # IR
        df.at[index, "informationretrieval"] = answer_dict1[row["informationretrieval"]]

        # statistics
        df.at[index, "statistics"] = answer_dict2[row["statistics"]]

        # databases
        df.at[index, "databases"] = answer_dict3[row["databases"]]

        # gender
        df.at[index, "gender"] = answer_dict4[row["gender"]]

        # chocolate
        df.at[index, "chocolate"] = answer_dict5[row["chocolate"]]

        # stand up
        df.at[index, "stand"] = answer_dict0[row["stand"]]

    # print(df["stand"].to_string())
    return df

# def programme(df):
#     AI_list = ["AI"]
#     Econometrics = [""]
#
#     for index, row in df.iterrows():
#         programme = row["programme"]
#         if any(word in programme for word in AI_list):
#             df.at[index, "programme"] = "AI"
#
#     return df

def make_ints_floats(df):
    for index, row in df.iterrows():

        # neighbors
        if not isinstance(row["neighbors"], int):
            df.at[index, "neighbors"] = "NaN"

        # stresslevel
        if not isinstance(row["stresslevel"], int):
            df.at[index, "stresslevel"] = "NaN"
        elif row["stresslevel"] < 0 or row["stresslevel"] > 100:
            df.at[index, "stresslevel"] = "NaN"

        # randomnumber
        if not (isinstance(row["randomnumber"], float) or isinstance(row["randomnumber"], int)):
            df.at[index, "randomnumber"] = "NaN"
        elif row["randomnumber"] >= 0 and row["randomnumber"] <= 10:
            df.at[index, "randomnumber"] = float(row["randomnumber"])
        else:
            df.at[index, "randomnumber"] = "NaN"



    # print(df["randomnumber"].to_string())
    return df

def money(df):

    for index, row in df.iterrows():
        if isinstance(row["money"], str):
            df.at[index, "money"] = "NaN"
        elif row['money'] < 0 or row['money'] > 100:
            df.at[index, "money"] = "NaN"
        else:
            df.at[index, "money"] = float(row['money'])

    return df

def lateness_bedtime(df_new):

    lateness_bedtime = []
    dict = {19: 0, 20:1, 21:2, 22:3, 23:4, 0:5, 1:6, 2:7, 3:8, 4:9, 5:10, 6:11, 7:12}
    for index, row in df_new.iterrows():
        bedtime = row["bedtime"]
        try:
            bedtime1 = bedtime.strftime("%H:%M:%S")[0:-3]
            bedtime1 = bedtime1.replace(":",".")

            h = int(bedtime1[:len(bedtime1)-3])

            bedtime2= bedtime.strftime("%M")
            bedtime2 = bedtime2.replace(":",".")
            new_min =  round(float(bedtime2)/60, 2)
            new_timestamp = float(str(dict[h]) + str(new_min)[1:len(str(new_min))])
            lateness_bedtime.append(new_timestamp)

        except:
            df_new.at[index, "bedtime"] = "NaN"
            lateness_bedtime.append("NaN")

    df_new['lateness_bedtime'] = lateness_bedtime
    print(df["lateness_bedtime"].to_string())

    return df

if __name__ == "__main__":
    dfold = pd.read_excel('data/ODI-2020_cleaned.xlsx')
    df = new_column_names(dfold)

    money = False

    # als we willen werken met money, zet money op true
    if money:
        df = money(df)
    else:
        df = df.drop(['money'], axis=1)

    # update for birthyear
    df = birthyears(df)

    # updates all questions with multiple choice answers
    df = MC(df)

    # makes ints and floats
    df = make_ints_floats(df)

    # fix bedtime
    df = lateness_bedtime(df)

    df.to_excel("all_cleaned.xlsx",sheet_name='clean')
