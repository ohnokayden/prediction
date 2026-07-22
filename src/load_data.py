# load_data,py serves to collate and clean the data from the 05/06 to 25/26 season
# ENSURE CMD IS ONLY RUN FROM THE ROOT DIR
import pandas as pd

from pathlib import Path

dir = Path("data/")
histories = []
for file in dir.iterdir():
    df = pd.read_csv(file, low_memory=False)
    series = pd.Series(Path(file).stem, index=range(380))
    df = pd.concat([df,series], axis=1)
    histories.append(df)
df = pd.concat(histories)
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
# df = df.reset_index(drop=True) # possible to remove this line-> allow to find different seasons-> reset form...
df.to_csv("history.csv", index=False)