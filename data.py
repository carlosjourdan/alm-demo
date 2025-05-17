import pandas as pd

folder = "global"

market_caps = pd.read_pickle(f"data/{folder}/market_caps.pkl")
cov_matrix = pd.read_pickle(f"data/{folder}/cov_matrix.pkl")
corr_matrix = pd.read_pickle(f"data/{folder}/corr_matrix.pkl")
risk_factors = list(market_caps.index)
