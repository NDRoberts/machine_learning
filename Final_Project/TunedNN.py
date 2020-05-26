#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Post-semester development of Neural Network regression program.
    Based on a subset of submitted project code, only stuff I wrote.
    Other code omitted for development/testing convenience.
    Gradually adapting to be more data-agnostic, fewer hard-coded values.
    @author: Nathaniel Roberts '''

import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import kerastuner as kt
from sklearn.model_selection import KFold, train_test_split
from matplotlib import pyplot as plt


def loadData(to_load=None):
    ''' Read a .CSV file into memory and return it as a Pandas dataframe '''
    if to_load == None:
        to_load = './ne15.csv'
    data = pd.read_csv(to_load)
    return data


def request_targets(data):
    ''' Specify target variables for regression analysis; if user declines to
        specify, use existing project spec as defaults '''
    targets = [['NO3', 'SO4', 'Cl'], 'NH4']
    specify = input("Would you like to specify comparison variables? (Y/N)")
    if specify.lower() == 'y':
        satisfied = {'indep': False, 'dep': False}
        while not all(satisfied.values()):
            if not satisfied['indep']:
                exp_list = input("Please list any number of explanatory", 
                                 "(independent) variables by name,",
                                 "separated by commas(','):")
                if not exp_list:
                    print("You must specify at least one independent variable.")
                    continue
                explanatories = exp_list.split(sep=',')
                for v in explanatories:
                    if v not in data.columns:
                        print(f"I'm sorry, but {v} is not the name of a",
                               "variable in the provided data.")
                        continue
                satisfied['indep']
                targets[0] = explanatories
            if not satisfied['dep']:
                explained = input("Please specify one dependent variable by name:")
                if not explained:
                    print("You must specify a dependent variable.")
                if  explained not in data.columns:
                    print(f"I'm sorry, but {explained} is not the name of a",
                           "variable in the provided data.")
                    continue
                satisfied['dep']
                targets[1] = explained
    return targets


def cropData(data, var_set):
        ''' Exclude data irrelevant to this operation, as well as entries
            for which relevant data is incomplete. 
            Argument var_set should be a list of two items:
            (1) A list of independent variables
            (2) A single dependent variable '''
        data = data.applymap(lambda x: np.NaN if x == -9 else x)
        varbles = var_set[0] + [var_set[1]]
        varnish = ""
        for v in var_set[0]:
            varnish = varnish + (f"{v} != 'NaN' &")
        varnish += f"{var_set[1]} != 'NaN'"
        data = data.loc[:, varbles].query(varnish)
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


def hyper_build(hyparams):
    ''' Construct & compile a tunable hypermodel with given param options '''
    hmodel = Sequential(name='TunedModel')
    hmodel.add(tf.keras.layers.Input(shape=(3,)))
    for k in range(hyparams.Int('num_layers', 2, 20)):
        hmodel.add(Dense(
            name='dense_'+str(k),
            units=hyparams.Int(name='units_'+str(k), min_value=3, max_value=36, step=3),
            activation=hyparams.Choice('activation_' + str(k), ['relu', 'tanh', 'linear'])
        ))
    hmodel.add(Dense(name='bite_down', units=1, activation='linear'))
    hmodel.compile(
        optimizer=tf.keras.optimizers.Adadelta(
            learning_rate=hyparams.Choice('learning-rate_'+str(k),
                                          [1.0, 0.1, 1e-2, 1e-3, 1e-4]),
            rho=hyparams.Choice('rho_'+str(k), [0.90, 0.95, 0.98])),
        loss=hyparams.Choice('loss_'+str(k), 
                             ['mean_squared_error', 
                              'mean_squared_logarithmic_error', 'logcosh']),
        metrics=['mean_squared_error']
    )
    return hmodel


def hyper_tune(hyparams, xTrain, yTrain, xVal, yVal):
    ''' Given training and validation data, use Keras-Tuner to find best
        hyperparameters, and return a model with those hyperparameters '''
    # Note to self: 10 Hyperband iterations is maybe TOO MANY
    tuner = kt.tuners.Hyperband(
        hypermodel=hyper_build,
        objective='val_mean_squared_error',
        max_epochs=100,
        factor=2,
        hyperband_iterations=3,
        hyperparameters=hyparams,
        directory='tuner_data',
        project_name='Final',
        overwrite=False)
    
    # # Alternative RandomSearch tuner seems to run faster
    # tuner = kt.tuners.RandomSearch(
    #     hypermodel=hyper_build,
    #     objective='val_mean_squared_error',
    #     max_trials=5,
    #     seed=42,
    #     hyperparameters=hyparams,
    #     executions_per_trial=4,
    #     directory='tuner_data',
    #     project_name='the_actual',
    #     overwrite=True)
    
    tuner.search_space_summary()
    halt = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
    tuner.search(xTrain, yTrain, epochs=100, validation_data=(xVal, yVal),
                 callbacks=[halt])
    tuner.results_summary()
    # print(tuner.get_best_hyperparameters()[0].values)
    return (tuner.get_best_models()[0], tuner.get_best_hyperparameters()[0].values)


def frame_params(prams):
    ''' Convert a given dict of params into a Pandas dataframe 
        (also print them, at least for now) '''
    n = 0
    model_parameters = {}
    tuner_parameters = {}
    for var_name in prams.keys():
        if var_name == 'num_layers':
            n = prams[var_name]
        elif var_name[0:5] == 'tuner':
            vn = var_name.split('/')
            tuner_parameters[vn[0]] = prams[var_name]
        else:
            vn = var_name.split('_')
            if vn[0] not in model_parameters:
                model_parameters[vn[0]] = {}
            model_parameters[vn[0]][vn[1]] = prams[var_name]
    model_frame = pd.DataFrame.from_dict(model_parameters)
    print("Number of hidden layers used:", n, '\n')
    print("Model parameters (by layer):\n", model_frame, '\n')
    print("Tuner parameters:")
    for p in tuner_parameters.keys():
        print(p, "=", tuner_parameters[p])


def fullPlot(X_data, y_true, y_pred):
    ''' Plots all predictions vs. observed values; currently only works for
        default variable selection '''
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


if __name__ == '__main__':
    data = loadData()
    data = cropData(data, request_targets(data))
    xTest, xTrain, xVal, yTrain, yTest, yVal = splitData(data)
    hyparams = kt.HyperParameters()
    super_model, super_params = hyper_tune(hyparams, xTrain, yTrain, xVal, yVal)
    plot_model(super_model, to_file="./TunedModelStruct.png", show_shapes=True)
    super_preds = np.array(super_model.predict(xTest))
    fullPlot(xTest, yTest, super_preds)
    super_model.summary()
    frame_params(super_params)