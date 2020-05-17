# import nltk
# from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
# nltk.download('punkt')
import pandas as pd
import numpy as np
import csv
# from nltk.tokenize import sent_tokenize, word_tokenize
# import warnings
#
# warnings.filterwarnings(action = 'ignore')
#
# import gensim
# from gensim.models import Word2Vec


def make_df():
    dict= {"label":[], "text":[]}
    # df = pd.read_csv("data/SmsCollection.csv",sep=";")
    with open('data/SmsCollection.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=";")
        line_count = 0
        for row in csv_reader:
            # print(row)
            if line_count == 0:
                line_count +=1
                # print(f'Column names are {", ".join(row)}')
                continue
            line_count+=1
            dict["label"].append(row["label"])
            dict["text"].append(row["text"])

    df = pd.DataFrame.from_dict(dict)
    return df

def count_upperlowercase_length(df):
    capitals = []
    lowercase = []
    textlength = []

    for index, row in df.iterrows():
        text = row["text"]
        n_capitals = sum(1 for c in text if c.isupper())
        # print(text, n_capitals)
        n_lower = sum(1 for c in text if c.islower())
        length = sum(1 for c in text)
        capitals.append(n_capitals)
        lowercase.append(n_lower)
        textlength.append(length)

    df["capitals_n"] = capitals
    df["lowercase_n"] = lowercase
    df["textlength"] = textlength


    return df

def count_puncts(df):
    dict = {}

    # find all punction marks
    for index, row in df.iterrows():
        text = row["text"]
        for c in text:
            if not c.isalpha() and not c.isnumeric():

                # if punction mark already in dict, +1 in counter
                if c in dict:
                    dict[c][index] += 1
                else:

                    # if mark not yet in dict, add 0 list and +1 in counter
                    dict[c] = [0] * df.shape[0]
                    dict[c][index] +=1

    for key in dict.keys():
        df[key] = dict[key]
    return df

def run_all():
    df = make_df()
    df = count_upperlowercase_length(df)
    df = count_puncts(df)
    return df

def word_stemming(df):
    stemmer = PorterStemmer()

    # dirty_text = "He studies in the house yesterday, unluckily, the fans breaks down"
    def word_stemmer(words):
        stem_words = [stemmer.stem(o) for o in words]
        return " ".join(stem_words)

    for index, row in df.iterrows():
        text = row["text"]
        new_text = word_stemmer(text.split(" "))
        print("----------")
        print(text,new_text)
        print("----------")
    # clean_text1 = word_stemmer(dirty_text.split(" "))
    return df
def prob(df):
    # sample = open("C:\\Users\\Admin\\Desktop\\alice.txt", "r")
    # s = sample.read()
    # f = s.replace("\n", " ")

    data = []

    # iterate through each sentence in the file
    # for i in sent_tokenize(f):
    for index, row in df.iterrows():
        temp = []
        i = row["text"]

        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())

        data.append(temp)

    # Create CBOW model
    model1 = gensim.models.Word2Vec(data, min_count = 1,
                                  size = 100, window = 5)

    # print("Cosine similarity between 'dinner' " +
    #            "and 'birthday' - CBOW : ",
    # model1.similarity('dinner', 'birthday'))
    #
    # print("Cosine similarity between 'dinner' " +
    #            "and 'urgent' - CBOW : ",
    # model1.similarity('dinner', 'urgent'))


    # Create Skip Gram model
    model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100,
                                                 window = 5, sg = 1)

    # # Print results
    # print("Cosine similarity between 'dinner' " +
    #            "and 'birthday' - CBOW : ",
    # model2.similarity('dinner', 'birthday'))
    #
    # print("Cosine similarity between 'dinner' " +
    #            "and 'urgent' - CBOW : ",
    # model2.similarity('dinner', 'urgent'))

    return 1

if __name__ == "__main__":
    df = run_all()
    # word_stemming(df)
    # prob(df)
