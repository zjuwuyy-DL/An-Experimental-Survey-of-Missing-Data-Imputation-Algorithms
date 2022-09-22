import numpy as np

from MissingImputer import MissingImputer


X = np.array([
    [3,0, np.nan, np.nan],  # odd: implicit zero
    [2,5, np.nan, np.nan],  # odd: explicit nonzero
    [7,0, 0, np.nan],    # even: average two zeros
    [8,-5, 0, np.nan],   # even: avg zero and neg
    [2,0, 5, np.nan],    # even: avg zero and pos
    [1,4, 5, np.nan],    # even: avg nonzeros
    [-3,-4, -5, np.nan],  # even: avg negatives
    [0,-1, 2, np.nan],   # even: crossing neg and pos
]).transpose()



MissImputer = MissingImputer(ini_fill = True, model_reg = "xgboost", model_clf = "xgboost")

X_trans = MissImputer.fit(X).transform(X.copy())

print(X_trans)