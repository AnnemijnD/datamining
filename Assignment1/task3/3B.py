import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model, metrics, model_selection, preprocessing, tree, datasets
from scipy.spatial import distance


"""
Poging zonder train en test split
"""

df2 = pd.read_csv('weight-height.csv')

weight = df2['Weight'].values.reshape(-1, 1)
height = df2['Height'].values.reshape(-1, 1)

linear_r = linear_model.LinearRegression()
model = linear_r.fit(weight, height)
predictions_lr = linear_r.predict(height)

MSE = metrics.mean_squared_error(height, predictions_lr)
MAE = metrics.mean_absolute_error(height, predictions_lr)
print(MSE, MAE)
print(linear_r.coef_)


# range berekenen based on eucledian norm
# np.linalg.norm(a, b)

norms = []
for index, row in df2.iterrows():
    observed = (row['Weight'], row['Height'])
    predicted = (row['Weight'], predictions_lr[index])
    # norms.append(np.linalg.norm(observed, predicted))
    norms.append(distance.euclidean(observed, predicted))



print(min(norms))
print(max(norms))

# Decision tree
weight_tree = df2[['Weight']].values
height_tree = df2['Height'].values
sort_id_weight = weight_tree.flatten().argsort()
weight_tree = weight_tree[sort_id_weight]
height_tree = height_tree[sort_id_weight]

tree_m = tree.DecisionTreeRegressor(criterion='mse', max_depth=3)
tree_m.fit(weight_tree, height_tree)
predictions_tree = tree_m.predict(weight_tree)
MSE = metrics.mean_squared_error(height, predictions_tree)
MAE = metrics.mean_absolute_error(height, predictions_tree)
print(MSE, MAE)


tree_m = tree.DecisionTreeRegressor(criterion='mae', max_depth=3)
tree_m.fit(weight_tree, height_tree)
predictions_tree2 = tree_m.predict(weight_tree)
MSE = metrics.mean_squared_error(height, predictions_tree2)
MAE = metrics.mean_absolute_error(height, predictions_tree2)
print(MSE, MAE)


# HIER STAAT HET PLOTJE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ax = df2.plot.scatter(x='Weight', y = 'Height')
x = np.linspace(0, max(df2['Weight']), max(df2['Weight']))
plt.plot(linear_r.coef_[0][0] * x + linear_r.intercept_, c='red', label="Linear regression")
plt.plot(weight_tree, predictions_tree, c = 'yellow', label="Tree regression MSE")
plt.plot(weight_tree, predictions_tree2, c ='green', label="Tree regression MAE")

plt.xlim(min(df2['Weight']) -10, max(df2['Weight']) + 10)
plt.ylabel("Height in inches", fontsize=12)
plt.xlabel("Weight in pounds", fontsize=12)
plt.title("Three regression methods for weight and height data of 10000 people", fontsize=14)
plt.legend()
plt.show()


"""
Icecream plot
"""
x = [1, 2, 3, 4, 5]
y = []
for i in range(len(x)):
    y.append(0.75 * x[i] + 1)

plt.scatter(x, y)
plt.plot(x, y)
plt.xlabel("Scoops of icecream", fontsize = 12)
plt.ylabel("Price of icecream", fontsize = 12)
plt.show()



"""
Poging 2 met weight and height
"""
#
# df2 = pd.read_csv('weight-height.csv')
# ax = df2.plot.scatter(x='Weight', y = 'Height')
#
# train, test = model_selection.train_test_split(df2)
#
# weight_train = train['Weight'].values.reshape(-1,1)
# height_train = train['Height'].values.reshape(-1,1)
#
# weight_test = test['Weight'].values.reshape(-1,1)
# height_test = test['Height'].values.reshape(-1,1)
#
# # Linear regression
# linear_r = linear_model.LinearRegression()
# model = linear_r.fit(weight_train, height_train)
# predictions_lr = linear_r.predict(height_test)
#
# MSE = metrics.mean_squared_error(height_test, predictions_lr)
# MAE = metrics.mean_absolute_error(height_test, predictions_lr)
# print(MSE, MAE)
# print(linear_r.coef_)
#
# x = np.linspace(0, max(df2['Weight']), max(df2['Weight']))
# ax.plot(linear_r.coef_[0][0] * x + linear_r.intercept_)
# # plt.show()
#
#
# # Decision tree
#
#
#
# """
# example ermee puzzelen
# """
#
# boston = datasets.load_boston()            # Load Boston Dataset
# dfTEST = pd.DataFrame(boston.data[:, 12])      # Create DataFrame using only the LSAT feature
# dfTEST.columns = ['LSTAT']
# dfTEST['MEDV'] = boston.target                 # Create new column with the target MEDV
# # print(dfTEST.head())
#
# X = dfTEST[['LSTAT']].values                          # Assign matrix X --> wel reshapen
# y = dfTEST['MEDV'].values                             # Assign vector y --> niet reshapen
#
# # print(X)
# # print(y)
#
#
# weight_train_tree = train[['Weight']].values
# height_train_tree = train['Height'].values
# sort_id_weight = weight_train_tree.flatten().argsort()
# weight_train_tree = weight_train_tree[sort_id_weight]
# height_train_tree = height_train_tree[sort_id_weight]
#
# tree_m = tree.DecisionTreeRegressor(criterion='mse', max_depth=3)
# tree_m.fit(weight_train_tree, height_train_tree)
# predictions_tree = tree_m.predict(test[['Weight']].values)
# plt.plot(test[['Weight']].values, predictions_tree)
# plt.show()


# sort_idx = X.flatten().argsort()                  # Sort X and y by ascending values of X
# X = X[sort_idx]
# y = y[sort_idx]
#
# tree = DecisionTreeRegressor(criterion='mse',     # Initialize and fit regressor
#                              max_depth=3)
# tree.fit(X, y)



"""
Poging 1 met cereal
"""
#
# df = pd.read_csv('cereal.csv')
#
# print(df.head())
#
# ax = df.plot.scatter(x='sugars', y='calories')
# ax.plot()
# # plt.show()
#
# train, test = model_selection.train_test_split(df)
#
# sugars_train = train['sugars'].values.reshape(-1,1)
# calories_train = train['calories'].values.reshape(-1,1)
#
# sugars_test = test['sugars'].values.reshape(-1,1)
# calories_test = test['calories'].values.reshape(-1,1)
#
# linear_r = linear_model.LinearRegression()
# model = linear_r.fit(sugars_train, calories_train)
# predictions = linear_r.predict(calories_test)
#
# MSE = metrics.mean_squared_error(calories_test, predictions)
# MAE = metrics.mean_absolute_error(calories_test, predictions)
# print(MSE, MAE)
# print(linear_r.coef_)
#
# x = np.linspace(0, 15, 15)
# ax.plot(linear_r.coef_[0][0] * x + linear_r.intercept_)
# # ax.plot(sugars_test, predictions)
#
#
#
#
# ridge = linear_model.Ridge(alpha=0.5)
# ridge.fit(sugars_train, calories_train)
# ridge_prediction = ridge.predict(calories_test)
# MSE_ridge = metrics.mean_squared_error(calories_test, ridge_prediction)
# MAE_ridge = metrics.mean_absolute_error(calories_test, ridge_prediction)
# print(MSE_ridge, MAE_ridge)
# print(ridge.coef_)
#
# ax.plot(ridge.coef_[0][0] * x + ridge.intercept_)
#
# lasso = linear_model.Lasso(alpha=0.1)
# lasso.fit(sugars_train, calories_train)
# lasso_prediction = lasso.predict(calories_test)
# MSE_lasso = metrics.mean_squared_error(calories_test, lasso_prediction)
# MAE_lasso = metrics.mean_absolute_error(calories_test, lasso_prediction)
# print(MSE_lasso, MAE_lasso)
# print(lasso.coef_)
#
# ax.plot(lasso.coef_[0] * x + lasso.intercept_)
#
#
# #
# # poly = preprocessing.PolynomialFeatures(degree=2)
#
# #
# # clf = linear_model.LinearRegression()
# # # clf.fit(sugars_train_poly, calories_train)
# # clf.fit(calories_train_poly, sugars_train)
# # poly_prediction = clf.predict(calories_test_poly)
# # MSE_poly = metrics.mean_squared_error(calories_test, poly_prediction)
# # MAE_poly = metrics.mean_absolute_error(calories_test, poly_prediction)
# # print(MSE_poly, MAE_poly)
# # print(clf.coef_[0])
#
#
# poly = preprocessing.PolynomialFeatures(degree = 4)
#
# calories_test_poly = poly.fit_transform(calories_test)
# calories_train_poly = poly.fit_transform(calories_train)
#
# sugars_train_poly = poly.fit_transform(sugars_train)
# sugars_test_poly = poly.fit_transform(sugars_test)
#
# poly.fit(sugars_train_poly, calories_train)
# poly_lin = linear_model.LinearRegression()
# poly_lin.fit(sugars_train_poly, calories_train)
# poly_prediction = poly_lin.predict(calories_test_poly)
#
# MSE_poly = metrics.mean_squared_error(calories_test, poly_prediction)
# MAE_poly = metrics.mean_absolute_error(calories_test, poly_prediction)
# print(MSE_poly, MAE_poly)
# print(poly_lin.coef_[0])
#
# plt.plot(clf.coef_[0][0] * x + clf.coef_[0][1] * x + clf.coef_[0][2] * x + clf.intercept_)
# # plt.plot(sugars_test, poly_prediction)
#
# plt.show()
#
#
# # # want to predict calories
# # model = sm.OLS(df['calories'], df['sugars']).fit()
# # predictions = model.predict(df['calories'])
# # linear_r = linear_model.LinearRegression()
# #
# # sugars = df['sugars'].values.reshape(-1,1)
# # calories = df['calories'].values.reshape(-1,1)
# # model = linear_r.fit(sugars, df['calories'])
# # # model = linear_r.fit([df['sugars'], df['protein'], df['carbo']], df['calories'])
# # predictions = linear_r.predict(calories)
# #
# # print(predictions)
# #
# # # print(model.summary())
# #
# #
# # MSE = metrics.mean_squared_error(df['calories'], predictions)
# # MAE = metrics.mean_absolute_error(df['calories'], predictions)
# # print(MSE, MAE)
# #
# # print("halloooo")
# #
# #
# # logistic_r = linear_model.LogisticRegression()
# # model = logistic_r.fit(sugars, df['calories'])
# # predictions2 = logistic_r.predict(calories)
# #
# # print(predictions2)
# #
# # MSE = metrics.mean_squared_error(df['calories'], predictions2)
# # MAE = metrics.mean_absolute_error(df['calories'], predictions2)
# # print(MSE, MAE)
