import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def estimate_propensity(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].values
    y = df["treatment"].values
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    ps = model.predict_proba(X)[:, 1]
    return ps, model

def nearest_neighbor_match(ps: np.ndarray, treatment: np.ndarray, caliper: float = 0.05):
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]
    control_ps = ps[control_idx]
    pairs = []
    used_controls = set()
    for i in treated_idx:
        diff = np.abs(control_ps - ps[i])
        j_rel = np.argmin(diff)
        if diff[j_rel] <= caliper:
            j = control_idx[j_rel]
            if j not in used_controls:
                pairs.append((i, j))
                used_controls.add(j)
    return np.array(pairs)

def matched_ate(df: pd.DataFrame, pairs: np.ndarray):
    y_t = df.iloc[pairs[:, 0]]["outcome"].values
    y_c = df.iloc[pairs[:, 1]]["outcome"].values
    return float(np.mean(y_t - y_c))