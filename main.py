from src.data import generate_synthetic
from src.psm import estimate_propensity, nearest_neighbor_match, matched_ate
from src.uplift import SLearner, TLearner
from src.segments import cate_by_segment

def run():
    df = generate_synthetic(8000, 7)
    features = ["x1", "x2", "x3", "x4", "x5"]
    ps, _ = estimate_propensity(df, features)
    pairs = nearest_neighbor_match(ps, df["treatment"].values, 0.05)
    ate = matched_ate(df, pairs)
    X = df[features]
    T = df["treatment"]
    Y = df["outcome"]
    slearner = SLearner()
    slearner.fit(X, T, Y)
    uplift_s = slearner.predict_uplift(X)
    tlearner = TLearner()
    tlearner.fit(X, T, Y)
    uplift_t = tlearner.predict_uplift(X)
    cate_s = cate_by_segment(uplift_s, df["segment"]) 
    cate_t = cate_by_segment(uplift_t, df["segment"]) 
    print("Matched ATE:", round(ate, 4))
    print("Top segments (S-Learner):")
    for seg, val in cate_s.head(3).items():
        print(seg, round(val, 4))
    print("Top segments (T-Learner):")
    for seg, val in cate_t.head(3).items():
        print(seg, round(val, 4))

if __name__ == "__main__":
    run()