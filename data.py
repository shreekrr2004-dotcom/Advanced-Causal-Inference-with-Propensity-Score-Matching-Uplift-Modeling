import numpy as np
import pandas as pd

def generate_synthetic(n: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    x4 = rng.uniform(-1, 1, n)
    x5 = rng.binomial(1, 0.5, n)
    X = np.column_stack([x1, x2, x3, x4, x5])
    p_t = 1 / (1 + np.exp(-(0.3 * x1 + 0.6 * x2 - 0.4 * x3 + 0.5 * x4 + 0.8 * x5)))
    T = rng.binomial(1, p_t)
    base = -1.0 + 0.8 * x1 - 0.6 * x2 + 0.4 * x3 + 0.2 * x4 + 0.5 * x5
    tau = 0.2 + 0.3 * (x1 > 0).astype(float) + 0.15 * x2 + 0.1 * x5
    p_y = 1 / (1 + np.exp(-(base + T * tau)))
    Y = rng.binomial(1, p_y)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4", "x5"]) \
        .assign(treatment=T, outcome=Y)
    q = pd.qcut(df["x1"], 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    df["segment"] = q
    return df