# lab.py


import pandas as pd
import numpy as np
import io
from pathlib import Path
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def prime_time_logins(login: pd.DataFrame):
    t = pd.to_datetime(login["Time"])
    mask = t.dt.hour.between(16, 19, inclusive="both")
    
    prime_counts = (login.loc[mask].groupby("Login Id").size().to_frame("Time"))

    all_users = login["Login Id"].unique()
    return prime_counts.reindex(all_users, fill_value=0)


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def count_frequency(login: pd.DataFrame):
    end = pd.Timestamp("2024-01-31 23:59:00")
    agg = (login.assign(Time=pd.to_datetime(login["Time"]))
                 .groupby("Login Id")
                 .agg(first=("Time", "min"), count=("Time", "size")))
    days = (end - agg["first"]).dt.days
    return agg["count"] / days


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def cookies_null_hypothesis():
    return[1,2]
                         
def cookies_p_value(N):
    n = 250
    p0 = 0.04
    observed = 15 / n
    
    sims = np.random.binomial(n, p0, size=N) / n
    p_val = np.mean(sims >= observed)
    return p_val


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def car_null_hypothesis():
    return  [1,3,4,5]

def car_alt_hypothesis():
    return [2,6]

def car_test_statistic():
    return [1,4]

def car_p_value():
    return 4


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def superheroes_test_statistic():
    return [1,4]

def bhbe_col(heroes):
    hair = heroes["Hair color"].astype(str).str.lower()
    eyes = heroes["Eye color"].astype(str).str.lower()
    return hair.str.contains("blond") & eyes.str.contains("blue")

def superheroes_observed_statistic(heroes):
    bhbe = bhbe_col(heroes)
    good = heroes["Alignment"].astype(str).str.lower().eq("good")
    return float(good[bhbe].sum() / bhbe.sum())

def simulate_bhbe_null(heroes, N):
    bhbe = bhbe_col(heroes).to_numpy()
    good = heroes["Alignment"].astype(str).str.lower().eq("good").to_numpy().astype(int)
    n = good.size
    sims = np.empty(N)
    for i in range(N):
        shuffled = np.random.permutation(good)
        sims[i] = shuffled[bhbe].mean()
    return sims

def superheroes_p_value(heroes):
    obs = superheroes_observed_statistic(heroes)
    sims = simulate_bhbe_null(heroes, 100000)
    p_val = float((sims >= obs).mean())
    decision = "Reject" if p_val < 0.01 else "Fail to reject"
    return [p_val, decision]


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def diff_of_means(data: pd.DataFrame, col: str = 'orange') -> float:
    y = data.loc[data['Factory'] == 'Yorkville', col].mean()
    w = data.loc[data['Factory'] == 'Waco', col].mean()
    return float(abs(y - w))

def simulate_null(data: pd.DataFrame, col: str = 'orange') -> float:
    shuffled = data.copy()
    shuffled['Factory'] = np.random.permutation(shuffled['Factory'].to_numpy())
    return diff_of_means(shuffled, col)

def color_p_value(data: pd.DataFrame, col: str = 'orange') -> float:
    obs = diff_of_means(data, col)
    sims = [simulate_null(data, col) for _ in range(1000)]
    return float(np.mean(np.array(sims) >= obs))


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def ordered_colors():
    return  [('yellow', 0.0),
 ('orange', 0.048),
 ('red', 0.234),
 ('green', 0.461),
 ('purple', 0.979)]


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


    
def same_color_distribution():
    return (0.006, 'Reject')


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def perm_vs_hyp():
    return ['P','P','H','H','P']
