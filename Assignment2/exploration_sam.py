import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import drop_cols

sns.set()
sns.set_color_codes("pastel")


def plot(var, data):
    """
    Makes a plot for the booked fraction of a variable of the training data set.
    """
    var_names = {"promotion_flag": "Promotion", "random_bool": "Random",\
                "prop_review_score": "Review score", "prop_starrating": "Star rating",\
                "prop_brand_bool": "Large brand", "prop_location_score1": "Location score 1",\
                "prop_location_score2": "Location score 1"}

    sns.barplot(x=var, y="booking_bool", data=data, color="b",\
                capsize=.1, errwidth=1.1).tick_params(labelsize=18)

    # plot style properties
    ax = plt.gca()

    for ax in plt.gcf().axes:
        x = ax.get_xlabel()
        y = ax.get_ylabel()
        ax.set_xlabel(x, fontsize=20)
        ax.set_ylabel(y, fontsize=20)

    plt.title("Bookings for variable " + str(var), fontsize=22)
    plt.ylabel("Fraction")
    plt.xlabel(var_names[var])
    t = ax.title
    t.set_position([.5, 1.05])
    # plt.ylim([0, 1])
    plt.subplots_adjust(bottom=.15, left=.15)
    plt.savefig("results/booking_" + str(var) + ".png", bbox_inches="tight")

    plt.show()

def plot_2(var, data):
    """
    Makes a plot for the booked fraction of a variable of the training data set.
    """
    var_names = {"promotion_flag": "Promotion", "random_bool": "Random",\
                "prop_review_score": "Review score", "prop_starrating": "Star rating",\
                "prop_brand_bool": "Large brand", "prop_location_score1": "Location score 1",\
                "prop_location_score2": "Location score 1"}

    sns.barplot(x=var, y="booking_bool", hue="comp1_rate", data=data, color="b",\
                capsize=.1, errwidth=1.1).tick_params(labelsize=18)

    # plot style properties
    ax = plt.gca()

    for ax in plt.gcf().axes:
        x = ax.get_xlabel()
        y = ax.get_ylabel()
        ax.set_xlabel(x, fontsize=20)
        ax.set_ylabel(y, fontsize=20)

    plt.title("Bookings for variable " + str(var), fontsize=22)
    plt.ylabel("Fraction")
    plt.xlabel(var_names[var])
    t = ax.title
    t.set_position([.5, 1.05])
    # plt.ylim([0, 1])
    plt.subplots_adjust(bottom=.15, left=.15)
    plt.savefig("results/booking_" + str(var) + ".png", bbox_inches="tight")

    # plt.show()

if __name__ == "__main__":

    # load data
    df_train = pd.read_csv("data/training_set_VU_DM.csv")
    # df_train = pd.read_csv("data/training_short.csv")

    # drop variables
    data = drop_cols(df_train)

    """
    jointplots
    """
    sns.distplot(data.prop_location_score2)
    plt.show()
    data.prop_location_score2 = np.exp(data.prop_location_score2)
    sns.distplot(data.prop_location_score2)
    plt.show()
    quit()
    sns.jointplot(x="prop_location_score1", y="prop_location_score2", data=data)
    plt.show()


    # print(data.groupby([data["random_bool"], data["promotion_flag"]])["booking_bool"].count())
    # print(data.groupby([data["comp1_rate"], data["comp2_rate"]])["booking_bool"].count())

    variables = ["promotion_flag", "random_bool", "prop_review_score", "prop_starrating", "prop_brand_bool"]

    # make plots for each variable
    # for var in variables:
    #     plot(var, data)
    #
    # ######################################################################

    # sns.countplot("srch_id", data=data) # duurt heeeel lang
    # plt.show()

    ######################################################################
    # te veel bins, hoe dit verminderen?


    # filter hotels that were booked
    booked = data[data["booking_bool"] == 1]

    # try to find linear correlations: only click_bool correlates to booking_bool
    # print(data.corr()["booking_bool"])

    predictors = [c for c in data.columns if c not in ["booking_bool","click_bool"]]
    from sklearn.model_selection import cross_validate
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
    scores = cross_validate(clf, data[predictors], data['booking_bool'], cv=3)
    print(scores)

    # # countplot for countries that customers travel from
    # sns.countplot('visitor_location_country_id', data=booked, order=booked.visitor_location_country_id.value_counts().iloc[:10].index)
    # plt.title("Most prevalent visitor locations", fontsize=22)
    # plt.subplots_adjust(bottom=.15, left=.15)
    # plt.savefig("results/visitor_locations.png", bbox_inches="tight")
    # plt.show()
    #
    # # countplot for countries that customers travel to
    # sns.countplot('prop_country_id', data=booked, order=booked.prop_country_id.value_counts().iloc[:10].index)
    # plt.title("Most prevalent hotel locations", fontsize=22)
    # # plt.subplots_adjust(bottom=.15, left=.15)
    # plt.savefig("results/hotel_locations.png", bbox_inches="tight")
    # plt.show()

###"""""""""""""""""""""""""""""""""""""###############################"
    print(data.info())
