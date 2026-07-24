import pandas as pd

from elo import EloSystem

def rollingForm():
    # should output +- int 
    pass

def elo(df: pd.DataFrame):
    # return the elo results of the latest game
    elo = EloSystem()
    res = {}
    for row in df.itertuples():
        res = elo.process_match(home=row.HomeTeam, away=row.AwayTeam, result=row.FTR, season=row.season)
    return res

def squadValue():
    # rmb to account for inflation-> use relative values value/avg value
    # consider using a comparative value
    pass

def h2h():
    # take from prev szns -> past 5 h2h games? 
    pass