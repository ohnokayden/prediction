import pandas as pd

from pathlib import Path

dir = Path("data/")
histories = []
for file in dir.iterdir():
    df = pd.read_csv(file)
    histories.append(df)
df = pd.concat(histories)
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
# df = df.reset_index(drop=True) # possible to remove this line-> allow to find different seasons-> reset form...
df.to_csv("data/history.csv", index=False)