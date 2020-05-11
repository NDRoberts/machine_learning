#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:49:32 2020

@author: Ryan McCullough, Nathaniel Roberts
"""
import os
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
# from sklearn import test_train_split
from sklearn.model_selection import KFold


def loadData():
    cwpath = "/ne15.csv"
    dpath = os.getcwd() + cwpath

    print("Loading Data ...")
    data = np.array(pd.read_csv(dpath))

    return data


def splitData(data):
    kfolds = KFold(n_splits=5, shuffle=True)
    folded_data = []
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1:]
    for train_inds, test_inds in kfolds.split(X):
        this_fold = {"train_X": X.iloc[train_inds, :],
                        "train_y": y.iloc[train_inds, :],
                        "test_X": X.iloc[test_inds, :],
                        "test_y": y.iloc[test_inds, :]}
        folded_data.append(this_fold)
    return folded_data


def trainModel(data):
    model = Sequential()
    model.add(Dense(15, input_dim=3, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(optimizer='adagrad', loss='categorical_crossentropy')

    (xTest, xTrain, xVal, yTest, yTrain, yVal) = splitData(data)

    trained = model.fit(xTrain, yTrain)

    return trained, xTest, yTest


def predictions(tModel, xTest, yTest):
    yPred = np.array(tModel.predict(xTest))
    for i in range(len(yPred)):
        print("Actual value: " + yTest[i] + "\n")
        print("Predicted value: " + yPred[i] + "\n")


if __name__ == '__main__':
    data = loadData()
    (trainedModel, xTest, yTest) = trainModel(data)
    predictions(trainedModel, xTest, yTest)
