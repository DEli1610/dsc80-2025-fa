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
    return 3


def trick_bool():
    return [4, 4, 12]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def population_stats(df):
    total = len(df)

    num_nonnull = df.notna().sum()

    prop_nonnull = num_nonnull / total

    num_distinct = df.nunique(dropna=True)

    prop_distinct = num_distinct / num_nonnull

    result = pd.DataFrame({
        'num_nonnull': num_nonnull,
        'prop_nonnull': prop_nonnull,
        'num_distinct': num_distinct,
        'prop_distinct': prop_distinct
    })

    return result


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_common(df, N=10):
    result = pd.DataFrame(index=range(N))
    
    for col in df.columns:
        counts = df[col].value_counts(dropna=False)
        
        top_vals = counts.index.tolist()[:N]
        top_counts = counts.values.tolist()[:N]
        
        if len(top_vals) < N:
            top_vals += [np.nan] * (N - len(top_vals))
            top_counts += [np.nan] * (N - len(top_counts))
        
        result[f"{col}_values"] = top_vals
        result[f"{col}_counts"] = top_counts
    
    return result


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    name_col = powers.columns[0]
    power_cols = powers.columns[1:]
    
    power_df = powers[power_cols].astype(bool)
    
    power_counts = power_df.sum(axis=1)
    hero_with_most = powers.loc[power_counts.idxmax(), name_col]

    flight_col = None
    for col in power_cols:
        if col.lower() == "flight":
            flight_col = col
            break
    
    flyers = power_df[power_df[flight_col]]
    flyers_wo_flight = flyers.drop(columns=[flight_col])
    most_common_among_flyers = flyers_wo_flight.sum().idxmax()
    
    one_power_mask = power_df.sum(axis=1) == 1
    one_power_df = power_df[one_power_mask]
    most_common_among_one_power = one_power_df.sum().idxmax()
    
    return [hero_with_most, most_common_among_flyers, most_common_among_one_power]



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    return heroes.replace(
        ['-', '–', '—', 'Unknown', 'unknown', 'None', 'none', 'NaN', 'nan', '', ' ', -99, -99.0],
        np.nan
    )



# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def super_hero_stats():
    return [
        "Apocalypse",          
        "DC Comics",           
        "good",                
        "Marvel Comics",       
        "Dark Horse Comics",   
        "Spider-Man"           
    ]



# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def clean_universities(df):
    cleaned = df.copy()

    cleaned['institution'] = cleaned['institution'].str.replace('\n', ', ', regex=False)

    cleaned['broad_impact'] = pd.to_numeric(cleaned['broad_impact'], errors='coerce').fillna(0).astype(int)

    parts = cleaned['national_rank'].str.split(',', n=1, expand=True)
    cleaned['nation'] = parts[0].str.strip()
    cleaned['national_rank_cleaned'] = (
        parts[1].str.extract(r'(\d+)', expand=False).astype(int)
    )
    cleaned = cleaned.drop(columns=['national_rank'])

    cleaned['nation'] = cleaned['nation'].replace({
        'USA': 'United States',
        'UK': 'United Kingdom',
        'Czechia': 'Czech Republic'
    })

    cleaned['is_r1_public'] = (
        (cleaned['control'] == 'Public')
        & cleaned['city'].notna()
        & cleaned['state'].notna()
    )

    return cleaned


def university_info(cleaned):

    df = cleaned.copy()

    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['world_rank'] = pd.to_numeric(df['world_rank'], errors='coerce')
    qf_col = 'quality of faculty'
    if qf_col not in df.columns:

        if 'quality_of_faculty' in df.columns:
            qf_col = 'quality_of_faculty'
        else:
            df['quality of faculty'] = np.nan
            qf_col = 'quality of faculty'
    df[qf_col] = pd.to_numeric(df[qf_col], errors='coerce')

    state_counts = df.groupby('state')['institution'].count()
    eligible_states = state_counts[state_counts >= 3].index
    df_eligible = df[df['state'].isin(eligible_states)]
    state_with_lowest_mean_score = (
        df_eligible.groupby('state')['score']
        .mean()
        .idxmin()
    )

    top100 = df[df['world_rank'] <= 100]
    if len(top100) == 0:
        proportion_qf_top100_among_top100 = 0.0
    else:
        proportion_qf_top100_among_top100 = float((top100[qf_col] <= 100).mean())

    share_private_by_state = 1.0 - df.groupby('state')['is_r1_public'].mean()
    num_states_majority_private = int((share_private_by_state >= 0.5).sum())

    best_in_nation = df[df['national_rank_cleaned'] == 1]
    worst_of_best_row = best_in_nation.loc[best_in_nation['world_rank'].idxmax()]
    worst_best_institution = worst_of_best_row['institution']

    return [
        state_with_lowest_mean_score,
        proportion_qf_top100_among_top100,
        num_states_majority_private,
        worst_best_institution
    ]


