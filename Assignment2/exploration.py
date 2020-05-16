import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import drop_cols, display_df

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
    # df_train = pd.read_csv("data/training_set_VU_DM.csv")
    df_train = pd.read_csv("data/training_short.csv")

    # drop variables
    data = drop_cols(df_train)

    """ hotel prices """
    # sns.distplot(data.prop_location_score2)
    # plt.show()

    # transform to make linear?
    # data.prop_location_score2 = np.exp(data.prop_location_score2)
    # sns.distplot(data.prop_location_score2)
    # plt.show()

    data_outlier1 = data[np.abs(data.price_usd-data.price_usd.mean()) <= (3*data.price_usd.std())]
    data_outlier = data[np.abs(data.price_usd-data.price_usd.mean()) <= 2000]

    # sns.distplot(data_outlier.price_usd.values)
    # plt.show()
    # quit()
    # sns.jointplot(x="price_usd", y="prop_location_score2", data=data_outlier1)
    # plt.show()
    # quit()
    # sns.jointplot(x="prop_location_score1", y="prop_location_score2", data=data)
    # plt.show()


    # print(data.groupby([data["random_bool"], data["promotion_flag"]])["booking_bool"].count())
    # print(data.groupby([data["comp1_rate"], data["comp2_rate"]])["booking_bool"].count())

    """ make plots """
    variables = ["promotion_flag", "random_bool", "prop_review_score", "prop_starrating", "prop_brand_bool"]

    # make plots for each variable
    for var in variables:
        # plot(var, data)
        pass


    """ only booked rooms """

    # filter hotels that were booked
    booked = data[data["booking_bool"] == 1]

    # try to find linear correlations: only click_bool correlates to booking_bool
    # print(data.corr()["booking_bool"])

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
