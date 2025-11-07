# lab.py


from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def after_purchase():

    return ["NMAR", "MD", "MAR", "MAR", "MAR"]



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def multiple_choice():

    return ["MAR", "MAR", "MAR", "NMAR", "MCAR"]



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------



def first_round():
    payments_fp = Path('data') / 'payment.csv'
    payments = pd.read_csv(payments_fp)

    birth_year = pd.to_datetime(payments['date_of_birth'], errors='coerce').dt.year
    ages = 2024 - birth_year

    is_missing = payments['credit_card_number'].isna()


    ages_missing = ages[is_missing]
    ages_not_missing = ages[~is_missing]

    obs_stat = abs(ages_missing.mean() - ages_not_missing.mean())

    n_perms = 1000
    perm_stats = []


    is_missing_arr = is_missing.to_numpy()
    ages_arr = ages.to_numpy()

    for _ in range(n_perms):
        shuffled_missing = np.random.permutation(is_missing_arr)

        g1 = ages_arr[shuffled_missing]
        g2 = ages_arr[~shuffled_missing]

        stat = abs(np.nanmean(g1) - np.nanmean(g2))
        perm_stats.append(stat)

    perm_stats = np.array(perm_stats)
    p_value = np.mean(perm_stats >= obs_stat)

    decision = 'R' if p_value < 0.05 else 'NR'

    return [float(p_value), decision]



def second_round():

    payments_fp = Path('data') / 'payment.csv'
    payments = pd.read_csv(payments_fp)

    birth_year = pd.to_datetime(payments['date_of_birth'], errors='coerce').dt.year
    ages = 2024 - birth_year

    is_missing = payments['credit_card_number'].isna()

    ages_missing = ages[is_missing].to_numpy()
    ages_not_missing = ages[~is_missing].to_numpy()

    obs_stat, _ = stats.ks_2samp(
        ages_missing[~np.isnan(ages_missing)],
        ages_not_missing[~np.isnan(ages_not_missing)]
    )

    n_perms = 1000
    perm_stats = []

    is_missing_arr = is_missing.to_numpy()
    ages_arr = ages.to_numpy()

    for _ in range(n_perms):
        shuffled_missing = np.random.permutation(is_missing_arr)

        g1 = ages_arr[shuffled_missing]
        g2 = ages_arr[~shuffled_missing]

        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        perm_stat, _ = stats.ks_2samp(g1, g2)
        perm_stats.append(perm_stat)

    perm_stats = np.array(perm_stats)
    p_value = np.mean(perm_stats >= obs_stat)

    decision = 'R' if p_value < 0.05 else 'NR'
    final_conclusion = 'D' if decision == 'R' else 'ND'

    return [float(p_value), decision, final_conclusion]



# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def verify_child(heights):


    pvalues = {}

    for col in heights.columns:
        if col.startswith("child_"):
            missing_mask = heights[col].isna()

            fathers_missing = heights.loc[missing_mask, 'father'].dropna()

            fathers_not_missing = heights.loc[~missing_mask, 'father'].dropna()

            ks_res = stats.ks_2samp(
                fathers_missing,
                fathers_not_missing,
                alternative='two-sided',
                mode='auto'
            )

            pvalues[col] = float(ks_res.pvalue)

    return pd.Series(pvalues)




# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def cond_single_imputation(new_heights):

    father_bins = pd.qcut(new_heights['father'], 4, labels=False, duplicates='drop')


    group_means = (
        new_heights
        .groupby(father_bins)['child']
        .transform('mean')
    )

    imputed_child = new_heights['child'].fillna(group_means)

    return imputed_child


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):

    observed = child.dropna().to_numpy()

    if len(observed) == 0:
        raise ValueError("Keine beobachteten Werte vorhanden – kann nichts imputieren.")

    counts, bin_edges = np.histogram(observed, bins=10)

    total = counts.sum()
    if total == 0:
        raise ValueError("Histogramm hat keine Beobachtungen gezählt.")

    probs = counts / total

    imputed_vals = []
    for _ in range(N):
        bin_idx = np.random.choice(np.arange(len(counts)), p=probs)
        low = bin_edges[bin_idx]
        high = bin_edges[bin_idx + 1]
        val = np.random.uniform(low, high)
        imputed_vals.append(val)

    return np.array(imputed_vals)


def impute_height_quant(child):

    missing_mask = child.isna()
    n_missing = missing_mask.sum()

    if n_missing == 0:
        return child.copy()

    new_vals = quantitative_distribution(child, n_missing)

    imputed_series = child.copy()
    imputed_series.loc[missing_mask] = new_vals

    return imputed_series



# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def answers():
    mc_answers = [1, 2, 2, 1]
    websites = [
        "https://data.gov/robots.txt",        # allowed
        "https://www.instagram.com/robots.txt"  # permited
    ]

    return mc_answers, websites



