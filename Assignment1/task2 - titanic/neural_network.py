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

sns.set()

# load data
# train_data = pd.read_csv("data/train.csv")
# test_data = pd.read_csv("data/test.csv")
df_train, df_test = run_both()
data = pd.concat([df_train, df_test], axis=0, sort=True)

# ACCURACY DROPS LESS THAN 2 % IF EXCLUDING THESE GROUPS
# uninteresting=["Title_Master","Title_Noble","Title_Other","Embarked_Q"]
# data.drop(uninteresting, axis=1, inplace=True)


X_train = data[pd.notnull(data['Survived'])].drop(['Survived'], axis=1)
y_train = data[pd.notnull(data['Survived'])]['Survived']
X_test = data[pd.isnull(data['Survived'])].drop(['Survived'], axis=1)

def create_model(layer, act, act_out, opt='Adam', dr=0.0):
    """
    Creates neural network model with specified amount of layers and activation types.
    """

    # set random seed for reproducibility
    seed(42)
    tf.random.set_seed(42)
    lyrs  = layer
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


# make model
model = create_model([10], "linear", "sigmoid")
# print(model.summary())

# train model
training = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# calculate predictions for test dataset
df_test['Survived'] = model.predict(X_test)
print(df_test.head(10))
df_test['Survived'] = df_test['Survived'].apply(lambda x: round(x,0)).astype('int')
solution = df_test[['PassengerId', 'Survived']]
solution.to_csv("Neural_Network_Solution.csv", index=False)

quit()

# for testing amount of layers
layers = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
# activation = ["linear", "sigmoid", "relu", "softmax"]
activation = ["linear", "relu"]
act_outs = ["sigmoid", "sigmoid"]

runs = 30
for i, act in enumerate(activation):
    val_accs = []
    for layer in layers:
        acc_avg = []
        for run in range(runs):
            model = create_model(layer, act, act_outs[i])

            # train model on full train set, with 80/20 CV split
            training = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
            val_acc = np.mean(training.history['val_accuracy'])
            print("Run ", run, " - ", act + " activation - layer " + str(layer))
            acc_avg.append(val_acc)
            # print("\n%s: %.2f%%" % ('val_acc', acc_avg*100))

        # save average accuracy of runs
        val_accs.append(np.mean(acc_avg))
        print("accuracy: " + str(np.mean(acc_avg)))

    # plot line for each activation method
    plt.plot(layers, val_accs, label=act)

# plotting
plt.title("Accuracy of neural network model with different layers (N=" + str(runs) + ")")
plt.xlabel("Layers")
plt.xticks(np.arange(1, len(val_accs) + 1, 1))
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

# # plot accuracy of model over time
# plt.plot(training.history['accuracy'])
# plt.plot(training.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
