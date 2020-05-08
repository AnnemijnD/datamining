from preprocessing import run_both
from random_forrest import prediction
from neural_network import create_prediction, prepare_data_for_model
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat


# prepare data
df_test, X_train, y_train, X_test = prepare_data_for_model()
X_train = X_train.drop(columns=['PassengerId'])
X_test = X_test.drop(columns=['PassengerId'])
pass_id_test = df_test['PassengerId']

# compare classifiers
runs = 100

nn = []
rf = []

for i in range(runs):
    acc_nn = create_prediction(df_test, X_train, y_train, X_test)
    acc_rf = prediction(X_train, y_train, pass_id_test, X_test)

    nn.append(acc_nn)
    rf.append(acc_rf)

print("NN - RF")
print("Means: ", np.mean(nn), np.mean(rf))
print("SD: ", np.std(nn), np.std(rf))
print("Shapiro W & p-value:", stat.shapiro(nn), stat.shapiro(nn))
print("t-test: ")
print(stat.ttest_ind(nn, rf))

# plt.boxplot([nn, rf])
# plt.show()
