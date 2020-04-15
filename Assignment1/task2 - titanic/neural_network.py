import pandas as pd
import numpy as np
# from data_exploration import add_titles
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from IPython.display import display
from numpy.random import seed
# import tensorflow as tf
from preprocessing import run_both

# load data
# train_data = pd.read_csv("data/train.csv")
# test_data = pd.read_csv("data/test.csv")
df_train, df_test = run_both()
data = pd.concat([df_train, df_test], axis=0, sort=True)

# ACCURACY DROPS LESS THAN 2 % IF EXCLUDING THESE GROUPS
# uninteresting=["Title_Master","Title_Noble","Title_Other","Embarked_Q"]
# data.drop(uninteresting, axis=1, inplace=True)

# to view dataset
# def display_all(df):
#     with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
#         display(df)
# display_all(data.describe(include='all').T)

X_train = data[pd.notnull(data['Survived'])].drop(['Survived'], axis=1)
y_train = data[pd.notnull(data['Survived'])]['Survived']
X_test = data[pd.isnull(data['Survived'])].drop(['Survived'], axis=1)

def create_model(lyrs=[8], act='linear', opt='Adam', dr=0.0):

    # set random seed for reproducibility
    # seed(42)
    # tf.random.set_seed(42)

    model = Sequential()

    # create first hidden layer
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))

    # create additional hidden layers
    for i in range(1,len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))

    # add dropout, default is none
    model.add(Dropout(dr))

    # create output layer
    model.add(Dense(1, activation='sigmoid'))  # output layer
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

model = create_model()
print(model.summary())

# train model on full train set, with 80/20 CV split
training = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
val_acc = np.mean(training.history['val_accuracy'])
print("\n%s: %.2f%%" % ('val_acc', val_acc*100))

# summarize history for accuracy
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
