from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


def sim(a, b):

    sim = np.sum((r_ap - r_amean) * (r_bp - r_bmean)) / (np.sqrt(np.sum(r_ap - r_amean)**2) * np.sqrt(np.sum(r_bp - r_bmean)**2))

    return sim


def pred(a, p):

    pred = r_amean + np.sum(sim(a, b) * (r_bp - r_bmean)) / np.sum(sim(a, b))


# predictors = [c for c in data.columns if c not in ["booking_bool","click_bool","gross_bookings_usd","position"]]
#
# clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
# scores = cross_val_score(clf, data[predictors], data['booking_bool'], cv=3)
# # scores = cross_validate(clf, data[predictors], data['booking_bool'], cv=3)
# print(scores)
