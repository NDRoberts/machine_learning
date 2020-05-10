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
from sklearn import test_train_split


def loadData():
    cwpath = "/ne15.csv"
    dpath = os.getcwd() + cwpath

    print("Loading Data ...")
    data = np.array(pd.read_csv(dpath))

    return data


def splitData(data):
    #/todo


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
