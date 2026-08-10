import pandas as pd

df = pd.read_csv("data/raw/stolpersteine_berlin_raw.csv")
df.columns = df.columns.str.strip()   # fix leading/trailing whitespace in column names

print("Columns:", df.columns.tolist())
print()
print("Shape:", df.shape)
print()
print("Bezirk counts:")
print(df["Bezirk"].value_counts())
print()
print("Missing Bezirk rows:", df["Bezirk"].isna().sum())
print()
print("Virtueller-Stein unique values:", df["Virtueller-Stein"].unique())
