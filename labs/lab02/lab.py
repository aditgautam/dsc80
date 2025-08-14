# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def trick_me():
    data = [
        ['a', 'aa', 1],
        ['b', 'bb', 2],
        ['c', 'cc', 3],
        ['d', 'dd', 4],
        ['e', 'ee', 5]
    ]

    cols = ['Name', 'Name', 'Age']
    tricky_1 = pd.DataFrame(data, columns= cols)

    tricky_1.to_csv('tricky_1.csv', index=False)
    tricky_2 = pd.read_csv('tricky_1.csv')

    if tricky_1.columns.equals(tricky_2.columns):
        return 2
    else:
        return 3


def trick_bool():
    data = [
        [True, True, True, True],
        [True, False, True, False],
        [False, False, True, True],
        [True, True, True, False]
    ]

    cols = [True, True, False, False]

    bools = pd.DataFrame(data, columns=cols)

    s1 = bools[True].shape
    s2 = bools[[True, True, False, False]].shape
    # s3 = bools[[True, False]].shape

    choices = {
        (2,1): 1,
        (2,2): 2,
        (2,3): 3,
        (2,4): 4,
        (3,1): 5,
        (3,2): 6,
        (3,3): 7,
        (3,4): 8,
        (4,1): 9,
        (4,2): 10,
        (4,3): 11,
        (4,4): 12
    }

    return[choices[s1], choices[s2], 13]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def population_stats(df: pd.DataFrame):
    n_rows = len(df)

    num_nonnull = df.notna().sum()
    prop_nonnull = num_nonnull / n_rows
    num_distinct = df.nunique(dropna= True)

    denom = num_nonnull.replace(0, np.nan)
    prop_distinct = (num_distinct / denom).fillna(0.0)

    ndf = pd.DataFrame({
        'num_nonnull': num_nonnull.astype(int),
        'prop_nonnull' : prop_nonnull.astype(float),
        'num_distinct': num_distinct.astype(int),
        'prop_distinct': prop_distinct.astype(float)
    })

    return ndf.loc[df.columns]


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_common(df: pd.DataFrame, N=10):
    out = pd.DataFrame(index=range(N))

    for col in df.columns:
        vc = df[col].value_counts(dropna=True)
        top = vc.iloc[:N]

        vals = list(top.index)
        cnts = list(top.values)

        if len(vals) < N:
            pad = N - len(vals)
            vals += [np.nan] * pad
            cnts += [np.nan] * pad

        out[f"{col}_values"] = vals
        out[f"{col}_counts"] = cnts

    return out


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def super_hero_powers(powers: pd.DataFrame):
    hero_names = powers.iloc[:, 0]
    power_data = powers.iloc[:, 1:]
    num_powers = power_data.sum(axis=1)
    most_powers_hero = hero_names.iloc[num_powers.idxmax()]

    flyers = power_data[power_data["Flight"]]
    flyers = flyers.drop(columns=["Flight"])
    most_common_for_flyers = flyers.sum().idxmax()

    one_power_mask = num_powers == 1
    one_power_heroes = power_data[one_power_mask]
    most_common_one_power = one_power_heroes.sum().idxmax()

    return [most_powers_hero, most_common_for_flyers, most_common_one_power]


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    return heroes.replace(
        to_replace=['-', '', 'Unknown', 'unknown', 'UNKNOWN', 'NaN', 'None', 'null', '-99', '-99.0', -99, -99.0],
        value=np.nan
    )

# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def super_hero_stats():
    return ['Onslaught', 'George Lucas', 'bad', 'Marvel Comics', 'NBC - Heroes', 'Groot']


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------

def clean_universities(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if 'institution' in out.columns:
        out['institution'] = out['institution'].astype(str).str.replace('\n', ', ', regex=False)

    if 'broad_impact' in out.columns:
        out['broad_impact'] = pd.to_numeric(out['broad_impact'], errors='coerce').astype(int)

    nat = out['national_rank'].astype(str).str.split(',', n=1, expand=True)
    out['nation'] = nat[0].str.strip()
    out['national_rank_cleaned'] = pd.to_numeric(nat[1].str.strip(), errors='raise').astype(int)

    nation_map = {
        'Czechia': 'Czech Republic',
        'UK': 'United Kingdom',
        'U.K.': 'United Kingdom',
        'USA': 'United States of America',
        'U.S.': 'United States of America',
        'U.S.A.': 'United States of America',
        'United States': 'United States of America',
        'South Korea': 'Republic of Korea',
        'Russia': 'Russian Federation',
    }
    out['nation'] = out['nation'].replace(nation_map)

    nonnull_triplet = out[['control', 'city', 'state']].notna().all(axis=1)
    is_public = out['control'].astype(str).str.strip().str.lower().eq('public')
    out['is_r1_public'] = (nonnull_triplet & is_public).fillna(False)

    out = out.drop(columns=['national_rank'])

    return out


# Q7b — analysis --------------------------------------------------------------

def university_info(cleaned: pd.DataFrame):

    df = cleaned.copy()

    states_with_3p = (
        df[df['state'].notna()]
        .groupby('state')
        .filter(lambda g: len(g) >= 3)
    )
    state_lowest_mean = (
        states_with_3p.groupby('state')['score']
        .mean()
        .idxmin()
    )

    world_top100 = df[df['world_rank'] <= 100]
    prop_qof_top100 = (world_top100['quality_of_faculty'] <= 100).mean()

    share_private_by_state = (
        df[df['state'].notna()]
        .assign(is_private=lambda x: ~x['is_r1_public'].fillna(False))
        .groupby('state')['is_private']
        .mean()
    )
    num_states_half_private_or_more = int((share_private_by_state >= 0.5).sum())

    national_1 = df[df['national_rank_cleaned'] == 1]
    worst_among_best_idx = national_1['world_rank'].idxmax()
    worst_among_best_institution = df.loc[worst_among_best_idx, 'institution']

    return [
        state_lowest_mean,
        float(prop_qof_top100),
        int(num_states_half_private_or_more),
        str(worst_among_best_institution),
    ]

