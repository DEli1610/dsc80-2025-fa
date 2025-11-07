# lab.py


import pandas as pd
import numpy as np
import io
from pathlib import Path
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def prime_time_logins(login):
    df = login.copy()
    df['Time'] = pd.to_datetime(df['Time'])
    is_prime = (df['Time'].dt.hour >= 16) & (df['Time'].dt.hour < 20)
    df['prime'] = is_prime
    result = df.groupby('Login Id')['prime'].sum().to_frame('Time')

    return result



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def count_frequency(login):
    df = login.copy()
    df['Time'] = pd.to_datetime(df['Time'])
    
    today = pd.Timestamp("2024-01-31 23:59:00")

    grouped = df.groupby('Login Id').agg(
        total_logins=('Time', 'count'),
        first_login=('Time', 'min')
    )
    grouped['days_on_site'] = (today - grouped['first_login']).dt.days

    grouped['frequency'] = grouped['total_logins'] / grouped['days_on_site'].replace(0, 1)

    return grouped['frequency']


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def cookies_null_hypothesis():
    return [1,3]
                         
def cookies_p_value(N):
    n_cookies = 250        
    p_burnt = 0.04           
    observed_burnt = 15      

    sims = np.random.binomial(n_cookies, p_burnt, size=N)

    p_val = np.mean(sims >= observed_burnt)

    return p_val


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def car_null_hypothesis():
    return [3, 5]

def car_alt_hypothesis():
    return [2, 6]

def car_test_statistic():
    return [4]

def car_p_value (): 
    return 2


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------

def superheroes_test_statistic():
    return [1, 3]


def bhbe_col(heroes):
    bhbe = (
        heroes['Eye color'].str.contains('blue', case=False, na=False)
        &
        heroes['Hair color'].str.contains('blond', case=False, na=False)
    )
    return bhbe


def superheroes_observed_statistic(heroes):
    bhbe = bhbe_col(heroes)
    subset = heroes[bhbe]
    proportion = (subset['Alignment'].str.lower() == 'good').mean()
    return proportion


def simulate_bhbe_null(heroes, N):
    # BHBE-Maske
    bhbe = bhbe_col(heroes).to_numpy()

    # True wenn Alignment == good (case-insensitive)
    is_good = heroes['Alignment'].str.lower().eq('good').to_numpy()

    # Anzahl Zeilen
    n_rows = len(heroes)

    # N Zufalls-Permutation-Indizes (jede Zeile: 1 Permutation)
    idx = np.argsort(np.random.rand(N, n_rows), axis=1)

    # permutierte good-Labels
    shuffled = is_good[idx]

    # nur BHBE-Zeilen berücksichtigen → Anteil berechnen
    stats = shuffled[:, bhbe].mean(axis=1)

    return stats


def superheroes_p_value(heroes):
    # beobachtete Teststatistik: Anteil "good" unter BHBE im echten Datensatz
    obs = superheroes_observed_statistic(heroes)

    # Nullverteilung per Simulation (one-sided: "greater")
    sims = simulate_bhbe_null(heroes, 100_000)

    # p-Wert: Anteil der simulierten Werte, die >= observed sind
    p = (sims >= obs).mean()

    # Entscheidung bei alpha = 1%
    decision = 'Reject' if p < 0.01 else 'Fail to reject'

    return [p, decision]



# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def diff_of_means(data, col='orange'):
    x = data[col]
    Yorkville = (data['Factory'] == 'Yorkville')
    Waco = (data['Factory'] == 'Waco')
    return x[Yorkville].mean() - x[Waco].mean()


def simulate_null(data, col='orange'):
    shuffled_factory = np.random.permutation(data['Factory'].to_numpy())
    data_null = data.copy()
    data_null['Factory'] = shuffled_factory
    return diff_of_means(data_null, col=col)

def color_p_value(data, col='orange'):
    obs = abs(diff_of_means(data, col=col))
    sims = np.array([simulate_null(data, col=col) for _ in range(1000)])
    p = (sims >= obs).mean()
    return p


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def ordered_colors():
    ...


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


    
def same_color_distribution():
    ...


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def perm_vs_hyp():
    return ['P','P','H','P','P']

