import pandas as pd

from elo import EloSystem

def rollingForm(df: pd.DataFrame, date: pd.DatetimeIndex, HomeTeam: str, AwayTeam: str):
    # should output +- int 
    # pass in the date, HomeTeam, and AwayTeam
    # filter by the date, collect prev 5 home/away result
    homeGames = df[(df["date"] < date) & (df["HomeTeam"] == HomeTeam)]

    # include check for required min number of games

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

def empiricalBucketDiagnostic(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    n_buckets: int = 4,
) -> pd.DataFrame:
    """
    Bins `feature_col` into `n_buckets` equal-sized groups (by quantile) and
    reports the empirical outcome distribution in each bucket -- completely
    bypassing any fitted model.
 
    Use this any time a model's coefficients look suspicious: it tells you
    whether the raw data itself shows a sensible relationship (bug, if any,
    is in the model fit / feature construction) or looks scrambled (bug is
    upstream, e.g. in feature engineering or a home/away mixup).
 
    Parameters
    ----------
    df : DataFrame containing at least feature_col and target_col.
    feature_col : name of the numeric column to bucket (e.g. "elo_diff").
    target_col : name of the categorical outcome column (e.g. "FTR").
    n_buckets : number of quantile buckets (default 4 = quartiles).
 
    Returns
    -------
    DataFrame indexed by bucket range, columns = outcome classes, values =
    proportion of matches in that bucket with that outcome (rows sum to 1).
 
    Example
    -------
    >>> empirical_bucket_diagnostic(data, "elo_diff", "FTR", n_buckets=5)
    """
    working = df[[feature_col, target_col]].copy()
 
    working["_bucket"] = pd.qcut(
        working[feature_col], q=n_buckets, duplicates="drop"
    )
 
    result = (
        working.groupby("_bucket", observed=True)[target_col]
        .value_counts(normalize=True)
        .unstack()
        .round(3)
    )
 
    # Also report how many matches fall in each bucket -- small buckets
    # produce noisy empirical rates, worth knowing at a glance.
    result.insert(0, "n_matches", working.groupby("_bucket", observed=True).size())
 
    return result