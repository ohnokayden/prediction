from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import feats
from elo import EloSystem

df = pd.read_csv("data/history.csv", low_memory=False)

# cannot feed relevant teams only-> will affect league mean
# feed the elos thru a np array
eloSys = EloSystem()
le = LabelEncoder()
categories = ["H", "D", "L"]
le.fit(categories)
eloFeats = []
for row in df.itertuples():
    elos = eloSys.process_match(home=row.HomeTeam, away=row.AwayTeam, result=row.FTR, season=row.season)
    eloFeats.append(elos)
featDf = pd.DataFrame(eloFeats)
data = pd.concat([df, featDf], axis=1)
# only train on data after 2005 to allow the elo to stabilise