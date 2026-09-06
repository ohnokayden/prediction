from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import feats
from elo import EloSystem
from sklearn.metrics import log_loss, accuracy_score

df = pd.read_csv("data/history.csv", low_memory=False)

# cannot feed relevant teams only-> will affect league mean
# feed the elos thru a np array
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


train = feats[(feats["season"] > 2011) & (feats["season"] < 2023)]
test = feats[feats["season"] >= 2023]


# trainX = pd.concat([train["home_elo_pre_match"], train["away_elo_pre_match"]], axis=1)
# testX = pd.concat([test["home_elo_pre_match"], test["away_elo_pre_match"]], axis=1)
trainX = train[["elo_diff"]]
testX = test[["elo_diff"]]

trainY = train["FTR"]
testY = test["FTR"]


model = LogisticRegression()
model.fit(trainX,trainY)

probs = model.predict_proba(testX)
preds = model.predict(testX)
 
model_logloss = log_loss(testY, probs, labels=model.classes_)
model_acc = accuracy_score(testY, preds)
 
naive_probs = np.tile([1/3, 1/3, 1/3], (len(testY), 1))  # coin-flip 3-way baseline
naive_logloss = log_loss(testY, naive_probs, labels=model.classes_)
 
print(f"\nLogistic regression -- log loss: {model_logloss:.4f}, accuracy: {model_acc:.3f}")
print(f"Naive uniform baseline -- log loss: {naive_logloss:.4f}")
 