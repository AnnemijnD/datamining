import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from numpy.random import seed
import tensorflow as tf
import seaborn as sns
from preprocessing import run_both
from data_exploration import display_df

sns.set()
sns.set_color_codes("pastel")


def prepare_data_for_model():

    df_train, df_test = run_both()
    print(display_df(df_train))
    data = pd.concat([df_train, df_test], axis=0, sort=True)

    # ACCURACY DROPS LESS THAN 2 % IF EXCLUDING THESE GROUPS - did not improve score
    # uninteresting=["Title_Master","Title_Noble","Title_Other","Embarked_Q"]
    # data.drop(uninteresting, axis=1, inplace=True)

    X_train = data[pd.notnull(data['Survived'])].drop(['Survived'], axis=1)
    y_train = data[pd.notnull(data['Survived'])]['Survived']
    X_test = data[pd.isnull(data['Survived'])].drop(['Survived'], axis=1)

    return df_test, X_train, y_train, X_test


def create_model_testing(lyrs, act, act_out, opt='Adam', dr=0.0):
    """
    Creates neural network model with specified amount of layers and activation types.
    """

    # set random seed for reproducibility
    seed(42)
    tf.random.set_seed(42)
    # lyrs  = layer
    model = Sequential()

    # create first hidden layer
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))

    # create additional hidden layers
    for i in range(1,len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))

    # add dropout, default is none
    model.add(Dropout(dr))

    # create output layer
    model.add(Dense(1, activation=act_out))  # output layer
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def create_model(lyrs=[10], act="linear", opt='Adam', dr=0.0):
    """
    Creates neural network model with specified amount of layers and activation types.
    """

    # set random seed for reproducibility
    seed(42)
    tf.random.set_seed(42)
    # lyrs  = layer
    model = Sequential()

    # create first hidden layer
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))

    # create additional hidden layers
    for i in range(1, len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))

    # add dropout, default is none
    model.add(Dropout(dr))

    # create output layer
    model.add(Dense(1, activation="sigmoid"))  # output layer
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # train model
    training = model.fit(X_train, y_train, epochs=100, batch_size=30, validation_split=0.2, verbose=0)

    # plot accuracy of model over time
    plt.plot(training.history['accuracy'])
    plt.plot(training.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    return model


def create_prediction(df_test, X_train, y_train, X_test):

    # make model
    model = create_model_testing([12], "linear", "sigmoid")
    print(model.summary())

    # train model
    training = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=0)

    # calculate predictions for test dataset
    df_test['Survived'] = model.predict(X_test)
    print(df_test.head(10))
    df_test['Survived'] = df_test['Survived'].apply(lambda x: round(x,0)).astype('int')
    solution = df_test[['PassengerId', 'Survived']]
    solution.to_csv("results/NN_prediction.csv", index=False)


def model_testing():

    # for testing amount of layers
    layers = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    # activation = ["linear", "sigmoid", "relu", "softmax"]
    activation = ["linear", "relu"]

    runs = 20
    for i, act in enumerate(activation):
        val_accs = []
        for layer in layers:
            acc_avg = []
            for run in range(runs):
                model = create_model_testing(layer, act, "sigmoid")

                # train model on full train set, with 80/20 CV split
                training = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=0)
                val_acc = np.mean(training.history['val_accuracy'])
                print("Run ", run, " - ", act + " activation - layer " + str(layer))
                acc_avg.append(val_acc)

            # save average accuracy of runs
            val_accs.append(round(np.mean(acc_avg)*100, 2))
            print("accuracy: " + str(np.mean(acc_avg)))

        # plot line for each activation method
        plt.plot(layers, val_accs, label=act)

    # plotting
    plt.title("Accuracy of neural network model with different layers (N=" + str(len(layers)) + ")", fontsize=22)
    plt.xlabel("Layers", fontsize=20)
    plt.xticks(np.arange(1, len(val_accs) + 1, 1), fontsize=18)
    plt.ylabel("Accuracy (%)", fontsize=20)
    plt.legend()
    plt.subplots_adjust(bottom=.15, left=.15)
    plt.savefig("results/linear-relu-" + str(runs) + "runs.png")
    plt.show()


def param_testing(X_train, y_train):

    model = KerasClassifier(build_fn=create_model, verbose=0)

    # define the grid search parameters
    batch_size = [10, 30, 50]
    epochs = [25, 50, 100]
    dr = [0.0, 0.2, 0.4]
    param_grid = dict(batch_size=batch_size, epochs=epochs, dr=dr)

    # search the grid
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        cv=2,
                        verbose=0)

    result = grid.fit(X_train, y_train)

    means = result.cv_results_['mean_test_score']
    stds = result.cv_results_['std_test_score']
    params = result.cv_results_['params']

    print(means)
    print(stds)
    print(params)


if __name__ == "__main__":

    df_test, X_train, y_train, X_test = prepare_data_for_model()
    # create_model()
    model_testing()
    # create_prediction(df_test, X_train, y_train, X_test)
    # param_testing(X_train, y_train)
