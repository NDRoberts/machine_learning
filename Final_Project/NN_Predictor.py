import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def vec_combine(v1, v2):
    result = []
    for x in v1:
        for y in v2:
            result.append((x, y))
    return result


class NN_Predictor:
    ''' Store a dataset, implement an MLPRegressor model with it, and
        record predictions. '''

    data = None
    model = None
    predictions = None
    folded_data = []

    def __init__(self, data=None):
        if not data:
            raise Exception("A data set must be supplied to create"
                            + "a Predictor.")
        # print(data)
        if data[-4:] not in [".csv", ".txt"]:
            data = f"{data}.csv"
        filename = f"/{data}"
        filepath = os.getcwd() + filename
        self.data = pd.read_csv(filepath)
        self.crop_and_regularize()
        self.split_data()

    def split_data(self, numfolds=5):
        self.kfolds = KFold(n_splits=numfolds, shuffle=True)
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1:]
        for train_inds, test_inds in self.kfolds.split(self.X):
            this_fold = {"train_X": self.X.iloc[train_inds, :],
                         "train_y": self.y.iloc[train_inds, :],
                         "test_X": self.X.iloc[test_inds, :],
                         "test_y": self.y.iloc[test_inds, :]}
            self.folded_data.append(this_fold)
        # print(self.folded_data[0]["train_X"])

    def crop_and_regularize(self):

        drop_cols = ["flagCa", "flagMg", "flagK", "flagNa", "flagNH4",
                     "flagNO3", "flagCl", "flagSO4", "flagBr", "valcode",
                     "invalcode"]
        self.data.drop(columns=drop_cols, inplace=True)
        self.data = self.data.applymap(lambda x: np.NaN if x == -9 else x)
        self.data = self.data.loc[:, ['NO3', 'SO4', 'Cl', 'NH4']]
        self.data = self.data.query('NO3 != "NaN" & SO4 != "NaN" & Cl != "NaN" & NH4 != "NaN"')
        scaler = StandardScaler()
        scaler.fit(np.array(self.data))
        self.std_data = pd.DataFrame(scaler.transform(self.data))

    def build(self):
        mlpParams = {
            "hidden_layer_sizes": vec_combine(
                [n for n in range(10, 101) if n % 10 == 0],
                [n for n in range(10, 101) if n % 10 == 0]
            ),
            "activation": ["logistic", "tanh", "relu"],
            "solver": ["sgd", "adam"],
            "alpha": [0.005, 0.001, 0.0001]
        }
        mlp = MLPRegressor(max_iter=1000)
        reg = GridSearchCV(estimator=mlp, param_grid=mlpParams, n_jobs=-1, scoring=make_scorer(mean_squared_error))
        reg.fit(X=self.X, y=np.ravel(self.y))
        print(reg.best_params_)


if __name__ == "__main__":
    print("Duh, I'm a brain machine.")
    bungus = NN_Predictor("NTN-NE15-w.csv")
    # print(bungus.folded_data[0]["train_X"])
    # print(bungus.folded_data[0]["train_y"])
    bungus.build()
