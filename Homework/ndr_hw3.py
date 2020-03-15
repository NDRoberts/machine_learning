import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


labels = []
animal_s2i = {
    "cat": 0,
    "dog": 1,
    "panda": 2
}
animal_i2s = {
    0: "cat",
    1: "dog",
    2: "panda"
}


def build_database(target):
    """ For each subdirectory in <target>:
        - Read each image as an array, resize to 32x32
        - Transform square matrix into a 3072x1 vector
        - Append animal_type truth value to the vector
        - Append vector to overall data set """
    data = []
    for cl in os.listdir(target):
        labels.append(cl)
        for name in os.listdir(f"{target}{cl}/"):
            img = cv2.imread(f"{target}{cl}/{name}")
            sml = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
            vec = np.ravel(sml)
            result = np.append(vec, animal_s2i[cl])
            data.append(result)
    return np.array(data)


def fold_and_test(arr):
    ''' Divide the total data set into 5 segments:
        (20%x4) 4 training & validation folds,
          (20%) 1 final testing set'''
    remainder, test = train_test_split(arr, test_size=0.2)
    remainder, fold4 = train_test_split(remainder, test_size=0.25)
    remainder, fold3 = train_test_split(remainder, test_size=0.3333)
    fold1, fold2 = train_test_split(remainder, test_size=0.5)
    split = [
        np.hsplit(fold1, [fold1.shape[1]-1]),
        np.hsplit(fold2, [fold2.shape[1]-1]),
        np.hsplit(fold3, [fold3.shape[1]-1]),
        np.hsplit(fold4, [fold4.shape[1]-1]),
        np.hsplit(test, [test.shape[1]-1])
    ]
    return split


def train_hyperparams(data):
    ''' Run KNN classification analysis using each value of hyperparameters '''
    test_k = [3, 5, 7]
    l_mode = [1, 2]
    scores = {}
    X_fold1, y_fold1 = data[0]
    X_fold2, y_fold2 = data[1]
    X_fold3, y_fold3 = data[2]
    X_fold4, y_fold4 = data[3]
    X_test, y_test = data[4]
    labels = [0, 1, 2]
    for k in test_k:
        for l in l_mode:
            model = KNeighborsClassifier(n_neighbors=k, p=l)
            # || For each training/validation fold, assemble its neighbors into
            #    a training group, then train & test against that fold.

            # \|/ Fold 1 as Validation:
            f1_trainer = np.vstack((X_fold2, X_fold3, X_fold4))
            f1_truth = np.vstack((y_fold2, y_fold3, y_fold4)).ravel()
            model.fit(f1_trainer, f1_truth)
            predicts1 = model.predict(X_fold1)
            aprf1 = (metrics.accuracy_score(y_fold1, predicts1),
                     metrics.precision_score(y_fold1, predicts1, average='macro', labels=labels),
                     metrics.recall_score(y_fold1, predicts1, average='macro', labels=labels),
                     metrics.f1_score(y_fold1, predicts1, average='macro', labels=labels))
            # \|/ Fold 2 as Validation:
            f2_trainer = np.vstack((X_fold1, X_fold3, X_fold4))
            f2_truth = np.vstack((y_fold1, y_fold3, y_fold4)).ravel()
            model.fit(f2_trainer, f2_truth)
            predicts2 = model.predict(X_fold2)
            aprf2 = (metrics.accuracy_score(y_fold2, predicts2),
                     metrics.precision_score(y_fold2, predicts2, average='macro', labels=labels),
                     metrics.recall_score(y_fold2, predicts2, average='macro', labels=labels),
                     metrics.f1_score(y_fold2, predicts2, average='macro', labels=labels))
            # \|/ Fold 3 as Validation:
            f3_trainer = np.vstack((X_fold1, X_fold2, X_fold4))
            f3_truth = np.vstack((y_fold1, y_fold2, y_fold4)).ravel()
            model.fit(f3_trainer, f3_truth)
            predicts3 = model.predict(X_fold3)
            aprf3 = (metrics.accuracy_score(y_fold3, predicts3),
                     metrics.precision_score(y_fold3, predicts3, average='macro', labels=labels),
                     metrics.recall_score(y_fold3, predicts3, average='macro', labels=labels),
                     metrics.f1_score(y_fold3, predicts3, average='macro', labels=labels))
            # \|/ Fold 4 as Validation:
            f4_trainer = np.vstack((X_fold1, X_fold2, X_fold3))
            f4_truth = np.vstack((y_fold1, y_fold2, y_fold3)).ravel()
            model.fit(f4_trainer, f4_truth)
            predicts4 = model.predict(X_fold4)
            aprf4 = (metrics.accuracy_score(y_fold4, predicts4),
                     metrics.precision_score(y_fold4, predicts4,
                                             average='macro', labels=labels),
                     metrics.recall_score(y_fold4, predicts4,
                                          average='macro', labels=labels),
                     metrics.f1_score(y_fold4, predicts4,
                                      average='macro', labels=labels))

            # || Now add average metric scores for (k, l) to dict
            scores[(k, l)] = (
                (aprf1[0] + aprf2[0] + aprf3[0] + aprf4[0]) / 4,
                (aprf1[1] + aprf2[1] + aprf3[1] + aprf4[1]) / 4,
                (aprf1[2] + aprf2[2] + aprf3[2] + aprf4[2]) / 4,
                (aprf1[3] + aprf2[3] + aprf3[3] + aprf4[3]) / 4
            )

    totals = dict.fromkeys(scores.keys(), 0)
    best = ((0, 0), 0)
    for r in totals:
        totals[r] = sum(scores[r])
        if totals[r] > best[1]:
            best = (r, totals[r])
    return (scores, best)


def final_test(model, X, y):

    print('buns')


def print_scores(dic):
    ''' Print out all scores in a trained parameter dictionary '''
    for k, l in dic:
        print(f"(K={k} || L={l})")
        print(f"Accuracy = {dic[(k, l)][0]} | Precision = {dic[(k, l)][1]}")
        print(f"Recall = {dic[(k, l)][2]}   | F-1 = {dic[(k, l)][3]}\n")


if __name__ == "__main__":
    base_directory = "./animals/"
    data = build_database(base_directory)
    split = fold_and_test(data)
    scores, best = train_hyperparams(split)
    print(f"The best scores overall were produced by {best[0]}:")
    print_scores(scores)

    # \\ And now, the piece de resistance:
    the_X_train = np.vstack((split[0][0], split[1][0], split[2][0], split[3][0]))
    the_y_train = np.vstack((split[0][1], split[1][1], split[2][1], split[3][1])).ravel()
    the_X_test = split[4][0]
    the_y_test = split[4][1].ravel()
    the_model = KNeighborsClassifier(n_neighbors=best[0][0], p=best[0][1])
    the_model.fit(the_X_train, the_y_train)
    the_predictions = the_model.predict(the_X_test)
    print("Classification Report of Final Results:")
    print(metrics.classification_report(the_y_test, the_predictions,
          target_names=labels))
