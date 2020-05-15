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
from preprocessing import run_both
from data_exploration import display_df

sns.set()
sns.set_color_codes("pastel")


def prepare_data_for_model():
    """
    Split the data into training, validation and test set.
    """

    df_train, df_test = run_both()
    # print(display_df(df_train))
    data = pd.concat([df_train, df_test], axis=0, sort=True)

    X_train = data[pd.notnull(data['Survived'])].drop(['Survived'], axis=1)
    y_train = data[pd.notnull(data['Survived'])]['Survived']
    X_test = data[pd.isnull(data['Survived'])].drop(['Survived'], axis=1)

    return df_test, X_train, y_train, X_test


def create_model_testing(lyrs, act, act_out, opt='Adam', dr=0.2):
    """
    Creates neural network model with specified amount of layers and activation types.
    """
    print("LYYYYYYYYYYYYYS", lyrs)
    # set random seed for reproducibility
    seed(42)
    tf.random.set_seed(42)

    # create sequential model
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

    # train model
    # training = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=0)

    # plot accuracy of model over time
    # plt.plot(training.history['accuracy'])
    # plt.plot(training.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    return model


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
    df_test['Survived'] = model.predict(X_test)
    df_test['Survived'] = df_test['Survived'].apply(lambda x: round(x, 0)).astype('int')
    solution = df_test[['PassengerId', 'Survived']]
    solution.to_csv("solutions/NN_prediction_relu_lay1-32_batch50_epoch25_dr2_dropout.csv", index=False)

    return val_acc


def model_testing():
    """
    Run models with various activation methods and amounts of layers.
    """

    # for testing amount of layers, each layer has 32 neurons
    layers = [[32], [32, 32], [32, 32, 32], [32, 32, 32, 32], [32, 32, 32, 32],\
            [32, 32, 32, 32, 32], [32, 32, 32, 32, 32, 32]]
    layers = [[1], [2], [4], [8], [16], [32], [64], [128]]

    # activation = ["linear", "sigmoid", "relu", "softmax"]
    activation = ["linear", "relu"]
    runs = 1
    for i, act in enumerate(activation):
        val_accs = []
        for layer in layers:
            acc_avg = []
            for run in range(runs):
                model = create_model_testing(layer, act, "sigmoid")

                # train model on full train set, with 80/20 CV split
                training = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)
                val_acc = np.mean(training.history['val_accuracy'])
                print("Run ", run, " - ", act + " activation - layer " + str(layer))
                acc_avg.append(val_acc)

            # save average accuracy of runs
            val_accs.append(round(np.mean(acc_avg)*100, 2))
            print("accuracy: " + str(np.mean(acc_avg)))

        # plot line for each activation method
        plt.plot([1,2,4,8,16,32,64,128], val_accs, label=act)

    # plotting
    plt.title("Accuracy of neural network model with different layers (N=" +\
            str(len(layers)) + ")", fontsize=22)
    plt.xlabel("Layers", fontsize=20)
    plt.xticks(np.arange(1, len(val_accs) + 1, 1), fontsize=18)
    plt.ylabel("Accuracy (%)", fontsize=20)
    plt.legend()
    plt.subplots_adjust(bottom=.15, left=.15)
    plt.savefig("results/32-linear-relu-" + str(runs) + "runs.png")
    plt.show()


def param_testing(X_train, y_train):
    """
    Hyperparameter tuning.
    """

    model = KerasClassifier(build_fn=create_model, verbose=0)

    # define the grid search parameters
    batch_size = [10, 891]
    epochs = [25, 50, 100]
    dr = [0.0, 0.2, 0.4]
    param_grid = dict(batch_size=batch_size, epochs=epochs, dr=dr)

    # search the grid
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        cv=5,
                        verbose=0)

    result = grid.fit(X_train, y_train)

    means = result.cv_results_['mean_test_score']
    stds = result.cv_results_['std_test_score']
    params = result.cv_results_['params']

    print(means)
    print(stds)
    print(params)

    print(grid.best_params_)
    print(f'Accuracy: {round(grid.best_score_*100, 2)}%')


if __name__ == "__main__":

    df_test, X_train, y_train, X_test = prepare_data_for_model()
    X_train = X_train.drop(columns=['PassengerId'])
    X_test = X_test.drop(columns=['PassengerId'])
    # create_model()
    model_testing()
    create_prediction(df_test, X_train, y_train, X_test)
    param_testing(X_train, y_train)
