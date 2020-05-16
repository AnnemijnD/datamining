import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from numpy.random import seed
import tensorflow as tf
import seaborn as sns
from preprocessing import prep_data

sns.set()
sns.set_color_codes("pastel")


def create_model(X_train, lyrs=[32], act="relu", opt='Adam', dr=0.2):
    """
    Creates neural network model with specified amount of layers and activation types.
    """

    # set random seed for reproducibility
    # seed(42)
    # tf.random.set_seed(42)

    model = Sequential()

    # create first hidden layer
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))

    # create additional hidden layers
    for i in range(1, len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))

    # dropout
    model.add(Dropout(dr))

    # create output layer
    model.add(Dense(1, activation="sigmoid"))  # output layer
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

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
    val_acc = np.mean(training.history['val_accuracy'])
    print("NN model validation accuracy during training: ", val_acc)

    # calculate predictions for test dataset
    df_test['category'] = model.predict(X_test)
    solution = df_test[['prop_id', 'srch_id', 'category']]

    # save prediction in output file
    sorted_sol = solution.sort_values(by='category', ascending=False)
    sorted_sol.drop("category", axis=1, inplace=True).to_csv("sorted2.csv", index=False)

    return val_acc

df_train = pd.read_csv("data/train_selection.csv")
# df_train = pd.read_csv("data/training_short.csv")
# df_train = pd.read_csv("data/training_set_VU_DM.csv")
# df_test = pd.read_csv("data/test_short.csv")
df_test = pd.read_csv("data/test_set_VU_DM.csv")

# preprocess data
data, df_test = prep_data(df_train, df_test)

# predicting columns of training set
predictors = [c for c in data.columns if c not in ["prop_id","srch_id","booking_bool",\
                            "click_bool","gross_bookings_usd","position","category"]]
X_train = data[predictors]

# predicting columns of test set
cols = [col for col in df_test.columns if col not in ['prop_id', 'srch_id']]
X_test = df_test[cols]

# prediction (outcome) variable
y_train = data.category.astype(int)


# create_model()
create_prediction(df_test, X_train, y_train, X_test)
