import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

class SLearner:
    def __init__(self):
        self.model = GradientBoostingClassifier()

    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series):
        X_aug = X.copy()
        X_aug["treatment"] = T.values
        self.model.fit(X_aug.values, Y.values)

    def predict_uplift(self, X: pd.DataFrame):
        X1 = X.copy()
        X0 = X.copy()
        X1["treatment"] = 1
        X0["treatment"] = 0
        p1 = self.model.predict_proba(X1.values)[:, 1]
        p0 = self.model.predict_proba(X0.values)[:, 1]
        return p1 - p0

class TLearner:
    def __init__(self):
        self.model_t = GradientBoostingClassifier()
        self.model_c = GradientBoostingClassifier()

    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series):
        Xt = X[T == 1]
        Xc = X[T == 0]
        Yt = Y[T == 1]
        Yc = Y[T == 0]
        self.model_t.fit(Xt.values, Yt.values)
        self.model_c.fit(Xc.values, Yc.values)

    def predict_uplift(self, X: pd.DataFrame):
        p1 = self.model_t.predict_proba(X.values)[:, 1]
        p0 = self.model_c.predict_proba(X.values)[:, 1]
        return p1 - p0