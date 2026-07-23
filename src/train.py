from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
import pandas as pd
import feats

df = pd.read_csv("data/history.csv", low_memory=False)

# cannot feed relevant teams only-> will affect league mean
elo = feats.elo(df) 
print(elo)

    