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
import tensorflow as tf
from preprocessing import run_both

# load data
# train_data = pd.read_csv("data/train.csv")
# test_data = pd.read_csv("data/test.csv")
df_train, df_test = run_both()
data = pd.concat([df_train, df_test], axis=0, sort=True)


# data = add_titles(data)
#
# data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
#
# # -------------------------------------------------------------
# # impute missing Age values using median of Title groups
# title_ages = dict(data.groupby('Title')['Age'].median())
# # create a column of the average ages
# data['age_med'] = data['Title'].apply(lambda x: title_ages[x])
# # replace all missing ages with the value in this column
# data['Age'].fillna(data['age_med'], inplace=True, )
# del data['age_med']
# # impute missing Fare values using median of Pclass groups
# class_fares = dict(data.groupby('Pclass')['Fare'].median())
# # create a column of the average fares
# data['fare_med'] = data['Pclass'].apply(lambda x: class_fares[x])
# # replace all missing fares with the value in this column
# data['Fare'].fillna(data['fare_med'], inplace=True, )
# del data['fare_med']
# data['Embarked'].fillna(method='backfill', inplace=True)
# # -------------------------------------------------------------
#
# # variables which need to be transformed to categorical
# to_categorical = ['Embarked', 'Title']
# print(data.head())
# for var in to_categorical:
#     data = pd.concat([data, pd.get_dummies(data[var], prefix=var)], axis=1)
#     del data[var]
#
# print(data.head())
#
# # convert to cateogry dtype
# data['Sex'] = data['Sex'].astype('category')
# # convert to category codes
# data['Sex'] = data['Sex'].cat.codes
#
# to_scale = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp']
# scaler = StandardScaler()
#
# for var in to_scale:
#     data[var] = data[var].astype('float64')
#     data[var] = scaler.fit_transform(data[var].values.reshape(-1, 1))
#
# def display_all(df):
#     with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
#         display(df)
# display_all(data.describe(include='all').T)

X_train = data[pd.notnull(data['Survived'])].drop(['Survived'], axis=1)
y_train = data[pd.notnull(data['Survived'])]['Survived']
X_test = data[pd.isnull(data['Survived'])].drop(['Survived'], axis=1)

def create_model(lyrs=[8], act='linear', opt='Adam', dr=0.0):

    # set random seed for reproducibility
    seed(42)
    tf.random.set_seed(42)

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
