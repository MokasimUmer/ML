import pandas as pd
df = pd.read_csv('insurance.csv')
print(df.shape)
print(df.describe())
df.isna().sum()
print(df.head())

# df = pd.read_csv("insurance.csv")  # your dataset, after encoding categorical vars

corr_matrix = df.corr(numeric_only=True)  # default = Pearson
print(corr_matrix["charges"])  # correlations of all features with charges
