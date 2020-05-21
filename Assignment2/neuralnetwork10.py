import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from numpy.random import seed
import tensorflow as tf
import seaborn as sns
import time
from preprocessing import prep_data
from keras.utils import np_utils


sns.set()
sns.set_color_codes("pastel")


def create_model(X_train, lyrs=[16], act="relu", opt='Adam', dr=0.2):
    """
    Creates neural network model with specified amount of layers and activation types.
    """

    # set random seed for reproducibility
    seed(42)
    tf.random.set_seed(42)

    model = Sequential()

    # create first hidden layer
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))

    # create additional hidden layers
    for i in range(1, len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))

    # dropout
    model.add(Dropout(dr))

    # create output layer
    model.add(Dense(3, activation="softmax"))  # output layer
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def create_prediction(df_test, X_train, y_train, X_test):
    """
    Create a prediction for the survival values of the testing set.
    """

    # make model: with or without dropout between hidden layers
    model = create_model(X_train)
    # model = create_dropout_model()
    # print(model.summary())

    # train model
    training = model.fit(X_train, y_train, epochs=25, batch_size=50,\
                        validation_split=0.2, verbose=0)
    print(training.history['val_accuracy'])
    val_acc = np.mean(training.history['val_accuracy'])
    print("NN model validation accuracy during training: ", val_acc)

    # calculate predictions for test dataset
    pd.DataFrame(model.predict(X_test)).to_csv("results/output_3classpred.csv", index=False)
    df_test[['category0','category1','category2']] = pd.DataFrame(model.predict(X_test))
    print(df_test.head())
    solution = df_test[['srch_id', 'prop_id', 'category']]

    date_time = time.strftime("%Y-%m-%d-%H-%M")
    # solution.to_csv("results/unsorted" + str(date_time) + ".csv", index=False)

    # save prediction in output file
    # solution = solution.sort_values(by='category', ascending=False)
    # solution = solution.drop("category", axis=1)
    solution.to_csv("results/solutions/nn_NIEUW-HOT_" + str(date_time) + ".csv", index=False)

    return val_acc


if __name__ == "__main__":
    # df_train = pd.read_csv("data/train_selection.csv")
    # df_train = pd.read_csv("data/training_set_VU_DM.csv")
    df_test = pd.read_csv("data/test_prep_NEWTEST.csv")
    df_train = pd.read_csv("data/training_prep_NEWTEST.csv")
    # df_test = pd.read_csv("data/test_set_VU_DM.csv")

    # preprocess data
    # df_train, df_test = prep_data(df_train, df_test)
    # df_test.to_csv("data/test_prep_long.csv", index=False)
    # df_train.to_csv("data/train_prep_long.csv", index=False)

    # predicting columns of training set
    predictors = [c for c in df_train.columns if c not in ["prop_id","srch_id","booking_bool",\
                                "click_bool","gross_bookings_usd","position", "category", 'Unnamed: 0']]
    X_train = df_train[predictors]
    # X_train.drop(["srch_id", "prop_id"], axis=1, inplace=True)

    # predicting columns of test set
    cols = [col for col in df_test.columns if col not in ['prop_id', 'srch_id', 'Unnamed: 0']]
    X_test = df_test[cols]
    # X_test.drop(["srch_id", "prop_id"], axis=1, inplace=True)

    # prediction (outcome) variable
    Y = df_train["category"]

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_train = np_utils.to_categorical(encoded_Y)
    # print(y_train.head())
    #
    # print(X_train.isnull().sum())
    # print(X_test.isnull().sum())

    """ functions """
    # create_model()
    # param_testing(X_train, y_train)
    # model_testing(X_train, y_train)
    create_prediction(df_test, X_train, y_train, X_test)
