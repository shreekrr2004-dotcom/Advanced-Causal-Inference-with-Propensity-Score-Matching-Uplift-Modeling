import numpy as np
import pandas as pd

def cate_by_segment(uplift: np.ndarray, segments: pd.Series):
    df = pd.DataFrame({"uplift": uplift, "segment": segments.values})
    g = df.groupby("segment")["uplift"].mean()
    return g.sort_values(ascending=False)