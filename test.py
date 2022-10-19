import pandas as pd

df = pd.read_csv("Dataku.csv", sep = ";")
df.drop(df.columns[[0]], axis = 1, inplace=True)

print(df.isnull().sum().sum())