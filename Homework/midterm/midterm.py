import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics


data = pd.read_table('./MNIST.csv', header=0,  delimiter=',')

log_reg_params = {'penalty': ['l2'], 'C': [0.1, 1, 10], 'solver': ['lbfgs', 'newton-cg']}
knn_params = {'n_neighbors': [3, 5, 7], 'p': [1, 2]}
svm_params = {'penalty': ['l1', 'l2'], 'loss': ['hinge'], 'C': [0.1, 1, 10]}

y_data = data.loc[:, 'label']
X_data = data.iloc[:, 1:]
y_train, y_test, X_train, X_test = train_test_split(y_data, X_data,
                                                    test_size=0.2, random_state=42)

scores = ['precision', 'recall', 'accuracy']

scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#-- Logistic Regression -- #
for s in scores:
    print(f"Scoring by {s}:")
    log_reg = LogisticRegression()
    log_reg_classifier = GridSearchCV(log_reg, log_reg_params, scoring=f"{s}_macro")
    log_reg_classifier.fit(X_train, y_train)
    print("Best parameters for Logistic Regression, based on cross-validation of training set:")
    print(log_reg_classifier.best_params_)
    lr_predictions = log_reg_classifier.predict(X_test)
    print(metrics.classification_report(y_test, lr_predictions))

# log_reg = LogisticRegression(solver='saga')
# log_reg.fit(X_train_scaled, y_train)
# predictions = log_reg.predict(X_valid_scaled)
# cnf_matrix = metrics.confusion_matrix(y_valid, predictions)
# print("Confusion Matrix:")
# print(cnf_matrix, '\n')
# fig, ax = plt.subplots()
# sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt='g')
# plt.title("Confusion Matrix")
# plt.ylabel("Actual")
# plt.xlabel("Predicted")
# plt.show()

# pltcolors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
#              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# plt.figure()
# y_pred_probabilities = np.array(np.arange(10))
# auc = np.array(np.arange(10))
# y_pred_probabilities = log_reg.predict_proba(X_train_scaled)
# fpr, tpr, _ = metrics.roc_curve(y_test_group,  y_pred_probabilities)
# auc[n] = metrics.roc_auc_score(y_test_group, y_pred_probabilities)
# plt.plot(fpr[n], tpr[n], label="data 1, auc="+str(auc[n]), color=pltcolors[n])
# plt.legend(loc=4)
# plt.show()

# -- KNN -- #
for s in scores:
    knn = KNeighborsClassifier()
    knn_classifier = GridSearchCV(knn, knn_params, scoring=f"{s}_macro")
    knn_classifier.fit(X_train, y_train)
    print("Best parameters for K Nearest Neighbors, based on cross-validation of training set:")
    print(knn_classifier.best_params_)
    knn_predictions = knn_classifier.predict(X_test)
    print(metrics.classification_report(y_test, knn_predictions))

# knn_model = KNeighborsClassifier(n_neighbors=5, p=1)
# knn_model.fit(X_train, y_train)
# knn_predictions = knn_model.predict(X_valid)
# print('\n', metrics.classification_report(y_valid, knn_predictions))

# -- SVM -- #
for s in scores:
    svm = LinearSVC()
    svm_classifier = GridSearchCV(svm, svm_params, score=f"{s}_macro")
    svm_classifier.fit(X_train, y_train)
    print("Best parameters for Support Vector Machine, based on cross-validation of training set:")
    print(svm_classifier.best_params_)
    svm_predictions = svm_classifier.predict(X_test)
    print(metrics.classification_report(y_test, svm_predictions))

# svm_model = LinearSVC(C=1, loss='hinge')
# svm_model.fit(X_train_scaled, y_train)
# svm_predictions = svm_model.predict(X_valid_scaled)
# plt.scatter(svm_predictions, y_valid)
# plt.plot(X_valid_scaled[:, 0], svm_predictions)
# plt.show()
# print('\n', metrics.classification_report(y_valid, svm_predictions))

# plt.figure()
# y_pred_probabilities = log_reg.predict_proba(x_test)[::, 1]
# fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_probabilities)
# auc = metrics.roc_auc_score(y_test, y_pred_probabilities)
# plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
# plt.legend(loc=4)
# plt.show()

# if y_data is not None:
#     print(y_data)
# if X_data is not None:
#     print(X_data.iloc[0, :])
