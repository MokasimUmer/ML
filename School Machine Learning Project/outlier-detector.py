import numpy as np
import pandas as pd

df = pd.read_csv('insurance.csv')
# Example: on 'charges' column of insurance df
Q1 = df['charges'].quantile(0.25)
Q3 = df['charges'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['charges'] < lower_bound) | (df['charges'] > upper_bound)]
print("Number of outliers:", len(outliers))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=["number"])
print(numeric_df.corr())