from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from feats import h2h
from elo import EloSystem
from sklearn.metrics import log_loss, accuracy_score

df = pd.read_csv("data/history.csv", low_memory=False)

# cannot feed relevant teams only-> will affect league mean
eloSys = EloSystem()
le = LabelEncoder()
categories = ["H", "D", "A"]

le.fit(categories)
eloFeats = []
for row in df.itertuples():
    elos = eloSys.process_match(home=row.HomeTeam, away=row.AwayTeam, result=row.FTR, season=row.season)
    eloFeats.append(elos)
featDf = pd.DataFrame(eloFeats)
data = pd.concat([df, featDf], axis=1)
data["elo_diff"] = (data["home_elo_pre_match"] - data["away_elo_pre_match"])

if {"WR","MP"}.issubset(data.columns):
    print("implemting h2h feat")
    data.drop(columns=["WR","MP"], inplace=True)
    wrMargin = []
    matchesPlayed = []
    for match in data.itertuples():
        record = h2h(df, match.Date, match.HomeTeam, match.AwayTeam, 10)
        wrMargin.append((record[0] - record[2])/ (record[0] + record[1] + record[2] + 1**-5))
        matchesPlayed.append(record[0] + record[1] + record[2])
    feats = {"WR": wrMargin, "MP": matchesPlayed}
    feats = pd.DataFrame(feats)
    data = pd.concat([data,feats], axis=1)

model = LogisticRegression(max_iter=1000)

train = data[(data["season"] > 2011) & (data["season"] < 2023)]
test = data[data["season"] >= 2023]

trainX = train[["WR","MP","elo_diff"]]
trainY = train["FTR"]

testX = test[["WR","MP","elo_diff"]]
testY = test["FTR"]

model = model.fit(trainX,trainY)

probs = model.predict_proba(testX)
preds = model.predict(testX)
 
model_logloss = log_loss(testY, probs, labels=model.classes_)
model_acc = accuracy_score(testY, preds)
 
naive_probs = np.tile([1/3, 1/3, 1/3], (len(testY), 1))  # coin-flip 3-way baseline
naive_logloss = log_loss(testY, naive_probs, labels=model.classes_)
 
print(f"\nLogistic regression -- log loss: {model_logloss:.4f}, accuracy: {model_acc:.3f}")
print(f"Naive uniform baseline -- log loss: {naive_logloss:.4f}")