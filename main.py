import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

filename = "Durham-Observatory-daily-climatological-dataset.xlsx"
df = pd.read_excel(filename,sheet_name = "Durham Obsy - daily data 1843-")
df.drop(df[df["YYYY"] < 2002].index, inplace=True)
df.drop(df.columns[14:], axis = 1, inplace=True)
dates = pd.to_datetime(dict(year=df["YYYY"], month=df["MM"], day=df["DD"]))
day_of_year = dates.dt.dayofyear
df.drop(columns = ["Date","DD","MM","YYYY"],inplace = True)
df.dropna(inplace=True)
year = 365.2425

df["Year sin"] = np.sin(day_of_year * (2 * np.pi / year))
df["Year cos"] = np.cos(day_of_year * (2 * np.pi / year))

n = len(df)
train_df = df[0:int(0.7*n)]
valid_df = df[int(0.7*n):int(0.9*n)]
test_df = df[int(0.9*n):]
n_features = df.shape[1]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(train_df)

columns=train_df.columns

trainNorm = pd.DataFrame(scaler.transform(train_df),columns=columns)
validNorm = pd.DataFrame(scaler.transform(valid_df),columns=columns)
testNorm = pd.DataFrame(scaler.transform(test_df),columns=columns)


print(train_df["Year sin"][57893])
print(train_df["Year sin"][57893+365])
