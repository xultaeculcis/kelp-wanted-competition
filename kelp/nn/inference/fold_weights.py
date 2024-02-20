import pandas as pd
from sklearn.preprocessing import MinMaxScaler

fold_scores = [
    ("fold=0", 0.7110),
    ("fold=1", 0.7086),
    ("fold=2", 0.7110),
    ("fold=3", 0.7139),
    ("fold=4", 0.7106),
    ("fold=5", 0.7100),
    ("fold=6", 0.7119),
    ("fold=7", 0.7105),
    ("fold=8", 0.7155),
    ("fold=9", 0.7047),
    ("fold=0v2", 0.7135),
    ("fold=1v2", 0.7114),
    ("fold=2v2", 0.7094),
    ("fold=3v2", 0.7133),
    ("fold=4v2", 0.7106),
]
df = pd.DataFrame(fold_scores, columns=["fold", "score"])

scaler = MinMaxScaler()
norm_scores = scaler.fit_transform(df[["score"]].values)
df["score_norm"] = norm_scores
df.to_parquet("scores.parquet")
