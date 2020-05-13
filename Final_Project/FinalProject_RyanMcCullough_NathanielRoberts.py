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
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold, train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


def loadData():
    ''' Read a .CSV file into memory and return it as a NumPy array '''
    cwpath = "/ne15.csv"
    dpath = os.getcwd() + cwpath

    print("Loading Data ...")
    data = pd.read_csv(dpath)

    return data


def cropData(data):
    ''' Extract only relevant columns, and only rows for which relevant
        data is complete '''
    drop_cols = ["flagCa", "flagMg", "flagK", "flagNa", "flagNH4",
                     "flagNO3", "flagCl", "flagSO4", "flagBr", "valcode",
                     "invalcode"]
    data.drop(columns=drop_cols, inplace=True)
    data = data.applymap(lambda x: np.NaN if x == -9 else x)
    data = data.loc[:, ['NO3', 'SO4', 'Cl', 'NH4']]
    data = data.query('NO3 != "NaN" & SO4 != "NaN"'
                        + '& Cl != "NaN" & NH4 != "NaN"')
    return data


def splitData(data, fold=0):
    ''' Divide data into Testing, Validation, and Training segments or,
        if a fold value is provided, into k cross-validation folds '''
    data = crop_and_regularize(data)
    print("Here")
    print(data)
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1:]
    if fold > 0:
        kfolds = KFold(n_splits=fold, shuffle=True)
        folded_data = []
        for train_inds, test_inds in kfolds.split(X):
            this_fold = {"train_X": X.iloc[train_inds, :],
                            "train_y": y.iloc[train_inds, :],
                            "test_X": X.iloc[test_inds, :],
                            "test_y": y.iloc[test_inds, :]}
            folded_data.append(this_fold)
        return folded_data
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8)
    train_X, validate_X, train_y, validate_y = train_test_split(train_X,
                                                                train_y,
                                                                train_size=0.75)
    return (test_X, train_X, validate_X, train_y, test_y, validate_y)


def trainModel(data):
    ''' Construct and train the artificial neural network model '''
    model = Sequential()
    model.add(Dense(15, activation='relu', input_shape=(3,)))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(optimizer='adagrad', loss='categorical_crossentropy')
    

    (xTest, xTrain, xVal, yTest, yTrain, yVal) = splitData(data)
    print(xTrain.shape)
    
    print(model.summary())

    trained = model.fit(xTrain, yTrain)

    return trained, xTest, yTest

def crop_and_regularize(data):
        ''' Exclude data irrelevant to this operation; also create
            standardized versions of the data I guess?  Not using them yet. '''
        drop_cols = ["flagCa", "flagMg", "flagK", "flagNa", "flagNH4",
                     "flagNO3", "flagCl", "flagSO4", "flagBr", "valcode",
                     "invalcode"]
        data.drop(columns=drop_cols, inplace=True)
        data = data.applymap(lambda x: np.NaN if x == -9 else x)
        data = data.loc[:, ['NO3', 'SO4', 'Cl', 'NH4']]
        data = data.query('NO3 != "NaN" & SO4 != "NaN"'
                                    + '& Cl != "NaN" & NH4 != "NaN"')
        scaler = StandardScaler()
        scaler.fit(np.array(data))
        std_data = pd.DataFrame(scaler.transform(data))
        return std_data


def predictions(tModel, xTest, yTest):
    ''' Make and print predictions using the trained model and test data '''
    yPred = np.array(tModel.predict(xTest))
    for i in range(len(yPred)):
        print("Actual value: " + yTest[i] + "\n")
        print("Predicted value: " + yPred[i] + "\n")

def plotit(xdata, *ydata):
    ''' Shorthand method to plot multiple data sets with shared X values
        (e.g. true y vs predicted y) '''
    for bit in ydata:
        plt.plot(xdata, bit, alpha=0.3)
    plt.show()


if __name__ == '__main__':
    data = splitData(cropData(loadData()))
    (trainedModel, xTest, yTest) = trainModel(data)
    predictions(trainedModel, xTest, yTest)
