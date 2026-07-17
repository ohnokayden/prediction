import numpy as np
import pandas as pd

# feature to add(rolling form) -> add features 1 by 1 and compare effect 
filename = "0506"
df_raw = pd.read_csv(f"../data/{filename}.csv") # loads respective year csv into a dataframe (df)
df_feats = pd.DataFrame()


# TODO: rolling home/away integration
# for the respective team, select by the team, slice last 6 games/view df[<home_team>][cur - 6 :cur]