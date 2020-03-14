import os
import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


animal_type = {
    "cat": 0,
    "dog": 1,
    "panda": 2
}


def build_database(target):
    """ For each subdirectory in <target>:
        - Read each image as an array, resize to 32x32
        - Transform square matrix into a 3072x1 vector
        - Append animal_type truth value to the vector
        - Append vector to overall data set """
    classes = []
    data = []
    for cl in os.listdir(target):
        classes.append(cl)
        for name in os.listdir(f"{target}{cl}/"):
            img = cv2.imread(f"{target}{cl}/{name}")
            sml = cv2.resize(img, (32, 32))
            vec = np.ravel(sml)
            result = np.append(vec, animal_type[cl])
            data.append(result)
    return np.array(data)


def test_and_folds(arr):
    remainder, test = train_test_split(arr, test_size=0.2)
    remainder, fold4 = train_test_split(remainder, test_size=0.25)
    remainder, fold3 = train_test_split(remainder, test_size=0.3333)
    fold1, fold2 = train_test_split(remainder, test_size=0.5)
    split = [
        np.hsplit(fold1, [fold1.shape[1]-1, -1]),
        np.hsplit(fold2, [fold2.shape[1]-1, -1]),
        np.hsplit(fold3, [fold3.shape[1]-1, -1]),
        np.hsplit(fold4, [fold4.shape[1]-1, -1]),
        np.hsplit(test, [test.shape[1]-1, -1])
    ]
    return split


base_directory = "./animals/"

biggole = build_database(base_directory)
print(f"My image database is now a {biggole.shape} array.")
splitsies = test_and_folds(biggole)
# manimal_hype = {
#     0: "cat",
#     1: "dog",
#     2: "panda"
# }
# for spoon in range(30):
#     print(f"Animal number {spoon * 100} is a {manimal_hype[biggole[spoon * 100][-1]]}!")
print("I now have 5 discrete data sets:")
for f in range(5):
    print(f"{f}. Size {splitsies[f][0].shape} plus {splitsies[f][1].shape}")
