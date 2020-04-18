import pandas as pd
import numpy as np
import dateparser
from datetime import datetime
from datetime import date as date_now

money_bool = False

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
                'What is your stress level (0-100)?': 'stress',
                'You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? ': 'money',
                'Give a random number': 'randomnumber',
                'Time you went to be Yesterday': 'bedtime',
                'What makes a good day for you (1)?': 'goodday1',
                'What makes a good day for you (2)?': 'goodday2'}

    df = df.rename(columns=column_dict)
    return df

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
    CLS = [" cls ", "computational science"]
    AI = [" ai ", "artificial intelligence"]
    BA = [" ba ", "business analytics"]
    CS = [" cs ", "computer science"]
    BF = [" bioinformatics "]
    econometrics = ["econometrics"] # '& operations research' wordt niet als losse master gezien nu
    QRM = [" qrm ", "quantitative risk management"]

    CLScounter = 0
    AIcounter = 0
    BAcounter = 0
    CScounter = 0
    BFcounter = 0
    Econcounter = 0
    QRMcounter = 0
    others = 0

    for index, row in df.iterrows():
        programme = " " + row["programme"].lower()+ " "

        if any(word in programme for word in CLS):
            df.at[index, "programme"] = "CLS"
            CLScounter +=1
        elif any(word in programme for word in AI):
            df.at[index, "programme"] = "AI"
            AIcounter +=1
        elif any(word in programme for word in BA):
            df.at[index, "programme"] = "BA"
            BAcounter +=1
        elif any(word in programme for word in CS):
            df.at[index, "programme"] = "CS"
            CScounter +=1
        elif any(word in programme for word in BF):
            df.at[index, "programme"] = "BF"
            BFcounter +=1
        elif any(word in programme for word in econometrics):
            df.at[index, "programme"] = "econometrics"
            Econcounter +=1
        elif any(word in programme for word in QRM):
            df.at[index, "programme"] = "QRM"
            QRMcounter +=1
        else:
            df.at[index, "programme"] = "other"
            others +=1

    # print("CLS: ", CLScounter, "AI", AIcounter, "BA ", BAcounter, "CS ", CScounter,
    #         "BF ", BFcounter, "EC ", Econcounter, "QRM", QRMcounter,"others", others)
    # one hot encoding
    df2 = pd.get_dummies(df["programme"],prefix='programme')
    # df = df.drop("programme", axis=1)
    df = pd.concat([df, df2], axis=1)

    return df




def MC(df):
    answer_dict0 = {"no": 0, "yes": 1, "unknown": "NaN"}
    answer_dict1 = {0: 0, 1: 1, "unknown": "NaN"}
    answer_dict2 = {"sigma": 0, "mu": 1, "unknown": "NaN"}
    answer_dict3 = {"nee": 0, "ja": 1, "unknown": "NaN"}
    answer_dict4 = {"male": 0, "female": 1, "unknown": "NaN"}



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

        # stand up
        df.at[index, "stand"] = answer_dict0[row["stand"]]

    return df

def make_ints_floats(df):
    for index, row in df.iterrows():

        # neighbors
        if not isinstance(row["neighbors"], int):
            df.at[index, "neighbors"] = "NaN"

        # stresslevel
        if not (isinstance(row["stress"], float) or isinstance(row["stress"], int)):
            df.at[index, "stress"] = "NaN"
        elif row["stress"] < 0 or row["stress"] > 100:
            df.at[index, "stress"] = "NaN"
        else:
            df.at[index, "stress"] = float(row["stress"])

        # randomnumber
        if not (isinstance(row["randomnumber"], float) or isinstance(row["randomnumber"], int)):
            df.at[index, "randomnumber"] = "NaN"
        elif row["randomnumber"] >= 0 and row["randomnumber"] <= 10:
            df.at[index, "randomnumber"] = float(row["randomnumber"])
        else:
            df.at[index, "randomnumber"] = "NaN"



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

def lateness_bedtime(df):

    lateness_bedtime = []
    dict = {19: 0, 20:1, 21:2, 22:3, 23:4, 0:5, 1:6, 2:7, 3:8, 4:9, 5:10, 6:11, 7:12}
    for index, row in df.iterrows():
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
            df.at[index, "bedtime"] = "NaN"
            lateness_bedtime.append("NaN")

    df['lateness_bedtime'] = lateness_bedtime
    df = df.drop("bedtime", axis=1)

    return df

def social_productive(df):

    df['social'] = len(df) * [0]
    df['productive'] = len(df) * [0]

    social_words = ["friend", "social", "family"]
    productive_words = ["productive", "study", "work", "working", "getting done",
        "productivity", "school", "research", "papers", "assignment", "coding",
        "goals", "achieve", "competing", "progress", "accomplish"]
    notwords = ["workout", "work out", "working out", "work-out"]

    for index, row in df.iterrows():

        good1 = str(row["goodday1"]).lower()
        good2 = str(row["goodday2"]).lower()

        if any(word in good1 for word in social_words):
            df.at[index, "social"] = 1

        if any(word in good2 for word in social_words):
            df.at[index, "social"] = 1

        if any(word in good1 for word in productive_words):
            if not any(word in good1 for word in notwords):
                df.at[index, "productive"] = 1

        if any(word in good2 for word in productive_words):
            if not any(word in good2 for word in notwords):
                df.at[index, "productive"] = 1

    # delete columns
    df = df.drop(columns=['goodday1', 'goodday2'])

    return df

def chocolate(df):
    # one hot encoding
    df = df.replace({'chocolate': "unknown"}, np.nan)
    df2 = pd.get_dummies(df["chocolate"],prefix='chocolate', dummy_na=True)
    # df = df.drop("chocolate", axis=1)

    df = pd.concat([df, df2], axis=1)
    return df


def run_all(money_bool):
    dfold = pd.read_excel('data/ODI-2020_cleaned.xlsx')
    df = new_column_names(dfold)

    # als we willen werken met money, zet money op true
    if money_bool:
        df = money(df)
    else:
        df = df.drop(['money'], axis=1)

    # update programme categories
    df = programme(df)
    df = chocolate(df)
    # update for birthyear
    df = birthyears(df)

    # updates all questions with multiple choice answers
    df = MC(df)

    # makes ints and floats
    df = make_ints_floats(df)

    # fix bedtime
    df = lateness_bedtime(df)

    df = social_productive(df)
    # df = df.drop("randomnumber", axis=1)
    df = df.replace('NaN', np.nan)
    df = df.replace('unknown', np.nan)


    return df


if __name__ == "__main__":

    df = run_all(True)

    df.to_excel("all_cleaned.xlsx",sheet_name='clean')
