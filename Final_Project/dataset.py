import numpy as np
import pandas as pd


data = pd.read_csv("./NTN-NE15-w.csv")
drop_cols = ["flagCa", "flagMg", "flagK", "flagNa", "flagNH4", "flagNO3",
             "flagCl", "flagSO4", "flagBr", "valcode", "invalcode"]
data.drop(columns=drop_cols, inplace=True)
data = data.applymap(lambda x: np.NaN if x == -9 else x)
elemqts_only = data.loc[:, ['NO3', 'SO4', 'Cl', 'NH4']]
complete_cases = elemqts_only.query('NO3 != "NaN" & SO4 != "NaN" & Cl != "NaN" & NH4 != "NaN"')

print(data.head())
print(elemqts_only.head())
print(complete_cases.head())
