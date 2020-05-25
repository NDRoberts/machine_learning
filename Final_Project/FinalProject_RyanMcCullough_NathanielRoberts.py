#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:49:32 2020

@author: Ryan McCullough, Nathaniel Roberts
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import kerastuner as kt
from sklearn.model_selection import KFold, train_test_split
from matplotlib import pyplot as plt


class MyperHodel(kt.HyperModel):

    # def __init__(self, name=None, tunable=True):
    #     pass

    def build(self, hyparams):
        model = Sequential()
        for k in range(hyparams.Int('layers', 2, 10)):
            model.add(Dense(
                units=hyparams.Int('units_' + str(k), min_value=10, max_value=100, step=10),
                activation=hyparams.Choice('activation_' + str(k), ['relu', 'tanh', 'linear'])))
        model.compile(
            optimizer='adadelta',
            loss='mean_squared_error',
            metrics='mean_squared_error')
        return model


def loadData():
    ''' Read a .CSV file into memory and return it as a Pandas dataframe '''
    cwpath = "/ne15.csv"
    dpath = os.getcwd() + cwpath
    print("Loading Data ...")
    data = pd.read_csv(dpath)
    return data


def cropData(data):
        ''' Exclude data irrelevant to this operation, as well as entries
            for which relevant data is incomplete '''
        drop_cols = ["flagCa", "flagMg", "flagK", "flagNa", "flagNH4",
                     "flagNO3", "flagCl", "flagSO4", "flagBr", "valcode",
                     "invalcode"]
        data.drop(columns=drop_cols, inplace=True)
        data = data.applymap(lambda x: np.NaN if x == -9 else x)
        data = data.loc[:, ['NO3', 'SO4', 'Cl', 'NH4']]
        data = data.query('NO3 != "NaN" & SO4 != "NaN" &'
                          + 'Cl != "NaN" & NH4 != "NaN"')
        return data


def splitData(data, fold=0):
    ''' Divide data into Testing, Validation, and Training segments or,
        if a fold value is provided, into k cross-validation folds '''
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1:]
    y = np.ravel(y)
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
    (xTest, xTrain, xVal, yTrain, yTest, yVal) = splitData(data)

    model = Sequential()
    model.add(Dense(15, activation='relu', input_shape=(3,)))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adagrad', loss='mean_squared_logarithmic_error')
    
    tf.keras.utils.plot_model(model, to_file="./ModelStruct.png", show_shapes=True)

    model.fit(xTrain, yTrain, epochs=10, validation_data=(xVal, yVal))
    
    yPred = np.array(model.predict(xTest))
    eval = model.evaluate(xTest, yTest, batch_size=128)
    print("Overall loss:", eval)
    print(model.summary())
    
    '''
    for i in range(len(yPred)):
        print("Actual value: ")
        print(yTest[i][0])
        print("\n")
        print("Predicted value: ") 
        print(yPred[i])
        print("\n")'''

    return (model, xTest, yTest, yPred)


def hyper_build(hyparams):
    hmodel = Sequential(name='TunedModel')
    # for k in range(hyparams.Int('layers', 2, 10)):
    hmodel.add(tf.keras.layers.Input(shape=(3,)))
    for k in range(1, 6):
        hmodel.add(Dense(
            name=f"Dense Layer {k}",
            units=hyparams.Int('units_' + str(k), min_value=3, max_value=21, step=3),
            activation=hyparams.Choice('activation_' + str(k), ['relu', 'tanh', 'linear'])
        ))
    hmodel.compile(
        optimizer='adadelta',
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )
    return hmodel


def hyper_tune(hyparams, xTrain, yTrain, xVal, yVal):
    model = MyperHodel()
    tuner = kt.tuners.Hyperband(
        hypermodel=model,
        objective='mean_squared_error',
        max_epochs=20,
        factor=2,
        hyperband_iterations=10,
        hyperparameters=hyparams,
        project_name='Final'
    )
    tuner.search(xTrain, yTrain, epochs=20, validation_data=(xVal, yVal))
    tuner.results_summary()
    return tuner.get_best_models()[0]


def fullPlot(X_data, y_true, y_pred):
    base = ['rh', 'gh', 'bh']
    elem = ['NO3', 'SO4', 'Cl']
    plt.figure(figsize=(14, 6))
    plt.suptitle("NH4 Concentrations: Observations v. Predictions")
    for n in range(3):
        ploc = 130 + (n + 1)
        plt.subplot(ploc)
        plt.xlabel(elem[n])
        plt.plot(X_data[:, n], y_true, base[n], alpha=0.3,
                 label=f"NH4 relative to {elem[n]} (observed)")
        plt.plot(X_data[:, n], y_pred, 'xk', alpha=0.3,
                 label=f"NH4 relative to {elem[n]} (predicted)")
        plt.legend()
    # plt.savefig("./results.png", format='png')
    plt.show()


def build_one_layer(hyparams):
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(3,)))
    model.add(Dense(
        units=hyparams.Int('units', min_value=3, max_value=36, step=3),
        activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adagrad(
            hyparams.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='logcosh',
        metrics=['mean_squared_error'])
    return model


if __name__ == '__main__':
    data = cropData(loadData())
    xTest, xTrain, xVal, yTrain, yTest, yVal = splitData(data)
    # (trainedModel, xTest, yTest, yPred) = trainModel(data)
    # fullPlot(xTest, yTest, yPred)
    hyparams = kt.HyperParameters()
    rstuner = kt.tuners.RandomSearch(
        hypermodel=build_one_layer,
        objective='val_mean_squared_error',
        max_trials=5,
        executions_per_trial=3,
        directory='tuner_data',
        project_name='tunedmodel'
    )
    rstuner.search_space_summary()
    rstuner.search(xTrain, yTrain, epochs=20, validation_data=(xVal, yVal))
    top2 = rstuner.get_best_models(num_models=2)
    prams = rstuner.get_best_hyperparameters()[0]
    rstuner.results_summary()
    print("Best 'units':", prams.get('units'),
          "   Best 'learning_rate':", prams.get('learning_rate'))
    # hmodel = hyper_tune(hyparams, xTrain, yTrain, xVal, yVal)
    # hmodel.fit(xTrain, yTrain)
    # hyperPreds = hmodel.predict(xTest)
    # print("The predictions coming back have shape", hyperPreds.shape)
    # heval = hmodel.evaluate(xTest, yTest, batch_size=64)
    # print("Tuned mean squared error:", heval)
    # print(hmodel.summary())
    # fullPlot(xTest, yTest, hyperPreds)