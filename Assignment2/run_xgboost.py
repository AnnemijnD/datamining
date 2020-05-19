import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost import DMatrix
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from multiprocessing import Pool
import pyltr

df_test = pd.read_csv("data/test_prep_long3-all.csv")
df_train = pd.read_csv("data/train_prep_long3-all.csv")

uninteresting = ["visitor_hist_starrating", "visitor_hist_adr_usd", "booking_bool", "click_bool"]
df_train.drop(uninteresting, axis=1, inplace=True)
uninteresting = ["visitor_hist_starrating", "visitor_hist_adr_usd"]
df_test.drop(uninteresting, axis=1, inplace=True)
# predictors = [c for c in df_train.columns if c not in ["prop_id","srch_id","booking_bool",\
#                                 "click_bool","gross_bookings_usd","position", "category", "visitor_hist_starrating", "visitor_hist_adr_usd"]]
# X_train = df_train[predictors]

    # predicting columns of test set
# cols = [col for col in df_test.columns if col not in ['prop_id', 'srch_id', "visitor_hist_starrating", "visitor_hist_adr_usd"]]
# X_test = df_test[cols]


def format(df, datatype):
    df = df.sort_values(by='srch_id', ascending=False)
    features = df[df.columns.tolist()].values
    qid = df['srch_id'].values
    if datatype == "train":
        target = df['category'].values
    else:
        target = np.zeros(len(df))

    return df, features, qid, target, df.columns.values.tolist()

# train
train, Xtr, qtr, ytr, feature_labels = format(df_train[df_train.srch_id % 10 != 0], "train")

# validation
vali, Xva, qva, yva, feature_labels = format(df_train[df_train.srch_id % 10 == 0], "train")

# test
test, Xte, qte, yte, feature_labels = format(df_test[df_test.srch_id % 10 == 0], "test")

comment = ' '.join(map(lambda t: '%d:%s' % t, zip(range(len(feature_labels)), feature_labels)))


def dump(args):
    # x, y, filename, query_id, comment = args
    dump_svmlight_file(*args, zero_based=False)

# p = Pool()
# p.map(dump, ((Xtr, ytr, 'data/train.svmlight', qtr, comment),
#              (Xva, yva, 'data/vali.svmlight', qva, comment),
#              (Xte, yte, 'data/test.svmlight', qte, comment)))
inp = [(Xtr, ytr, 'data/train.svmlight', qtr, comment),\
             (Xva, yva, 'data/vali.svmlight', qva, comment),\
             (Xte, yte, 'data/test.svmlight', qte, comment)]

for inpp in inp:
    dump(inp)
