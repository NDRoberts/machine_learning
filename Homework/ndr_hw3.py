import os
import cv2
import numpy as np
import pandas as pd
from enum import Enum
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
    return data


base_directory = "./animals/"

biggole = build_database(base_directory)
tanimal_ype = {
    0: "cat",
    1: "dog",
    2: "panda"
}
for spoon in range(30):
    print(f"Animal number {spoon * 100} is a {tanimal_ype[biggole[spoon * 100][-1]]}!")
