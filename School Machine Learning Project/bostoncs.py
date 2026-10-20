import pandas as pd

csv_url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

df = pd.read_csv(csv_url)
df.to_csv("boston_housing.csv", index=False)

print(df.head())
