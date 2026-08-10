import pandas as pd

ORIGINAL_DISTRICTS = [
    "Charlottenburg-Wilmersdorf",
    "Mitte",
    "Tempelhof-Schöneberg",
]

NEW_DISTRICTS = [
    "Friedrichshain-Kreuzberg",
    "Pankow",
    "Steglitz-Zehlendorf",
    "Neukölln",
]

df = pd.read_csv("data/raw/stolpersteine_berlin_raw.csv")
df.columns = df.columns.str.strip()

# Reproduce the ORIGINAL 3-district filter + ID assignment EXACTLY as the
# first version of this script did, so every already-scraped stolperstein_id
# stays unchanged. Only valid if stolpersteine_berlin_raw.csv is unchanged
# since Phase 1 (same row order, same content) -- confirm this before running.
original = df[df["Bezirk"].isin(ORIGINAL_DISTRICTS)].copy()
original = original.reset_index(drop=True)
original["stolperstein_id"] = original.index.map(lambda i: f"stolperstein_{i:05d}")

print("Original (3 districts) shape:", original.shape)

# New districts get FRESH ids continuing on from where the original set left
# off (stolperstein_07205 onward) -- never overlapping or reshuffling the
# existing 7205 ids.
start = len(original)
new = df[df["Bezirk"].isin(NEW_DISTRICTS)].copy()
new = new.reset_index(drop=True)
new["stolperstein_id"] = new.index.map(lambda i: f"stolperstein_{start + i:05d}")

print("New (4 districts) shape:", new.shape)
print(new["Bezirk"].value_counts())

combined = pd.concat([original, new], ignore_index=True)
print("Combined shape:", combined.shape)
print(combined["Bezirk"].value_counts())

combined.to_csv("data/raw/stolpersteine_7_districts.csv", index=False)
print("Saved to data/raw/stolpersteine_7_districts.csv")