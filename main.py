import pandas as pd

filename = "Durham-Observatory-daily-climatological-dataset.xlsx"
df = pd.read_excel(filename,sheet_name = "Durham Obsy - daily data 1843-")
df.drop(df[df["YYYY"] < 2002].index, inplace=True)
df.drop(df.columns[14:], axis = 1, inplace=True)

print(df.head())

