import pandas as pd

def change_sex(df):
    for index, row in df.iterrows():
        if row['Sex'] == 'male':
            df.at[index, 'Sex'] = 0
        else:
            df.at[index, 'Sex'] = 1

    return df

def run_all(df):

    df = change_sex(df)

    return df

def run_both():

    # load data
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")

    df_train = run_all(df_train)
    df_test = run_all(df_test)

    return df_train, df_test

if __name__ == "__main__":

    df_train, df_test = run_both()

    df_train.to_excel("train_processed.xlsx",sheet_name='clean')
    df_test.to_excel("test_processed.xlsx",sheet_name='clean')
