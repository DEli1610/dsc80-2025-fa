# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    cols = pd.Series(grades.columns)
    cols = cols.str.lower()
    
    def split(series): 
        return series.str.replace("_"," ").str.split().str.get(0)
   
    labs = split(cols[cols.str.contains("lab")])
    labs = list(dict.fromkeys(labs))
    project = split(cols[cols.str.contains("project")])
    project = list(dict.fromkeys(project))

    midterm = split(cols[cols.str.contains("midterm")])
    midterm = list(dict.fromkeys(midterm))

    final = split(cols[cols.str.contains("final")])
    final = list(dict.fromkeys(final))
    disc = split(cols[cols.str.contains("disc")])
    disc = list(dict.fromkeys(disc))
    checkpoint = split(cols[cols.str.contains("checkpoint")])
    checkpoint = list(dict.fromkeys(checkpoint))
   
    output = {
        "lab": labs,
        "project": project,
        "midterm": midterm,
        "final": final, 
        "disc": disc,
        "checkpoint": checkpoint
    }
    
    return output


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
     # Set variables for the calculation above and consider/replace na values with 0
     # Project 01
    earned_points_autograded_01 = grades["project01"].fillna(0)
    earned_points_free_response_01 = grades["project01_free_response"].fillna(0)
    max_points_autograded_01 = grades["project01 - Max Points"].fillna(0)
    max_points_free_response_01 = grades["project01_free_response - Max Points"].fillna(0)
    project_grade_01 = (earned_points_autograded_01 + earned_points_free_response_01) / (
        max_points_autograded_01 + max_points_free_response_01
    )

    # Project 02
    earned_points_autograded_02 = grades["project02"].fillna(0)
    earned_points_free_response_02 = grades["project02_free_response"].fillna(0)
    max_points_autograded_02 = grades["project02 - Max Points"].fillna(0)
    max_points_free_response_02 = grades["project02_free_response - Max Points"].fillna(0)
    project_grade_02 = (earned_points_autograded_02 + earned_points_free_response_02) / (
        max_points_autograded_02 + max_points_free_response_02
    )

    # Project 03
    earned_points_autograded_03 = grades["project03"].fillna(0)
    max_points_autograded_03 = grades["project03 - Max Points"].fillna(0)
    project_grade_03 = earned_points_autograded_03 / max_points_autograded_03

    # Project 04
    earned_points_autograded_04 = grades["project04"].fillna(0)
    max_points_autograded_04 = grades["project04 - Max Points"].fillna(0)
    project_grade_04 = earned_points_autograded_04 / max_points_autograded_04

    # Project 05
    earned_points_autograded_05 = grades["project05"].fillna(0)
    earned_points_free_response_05 = grades["project05_free_response"].fillna(0)
    max_points_autograded_05 = grades["project05 - Max Points"].fillna(0)
    max_points_free_response_05 = grades["project05_free_response - Max Points"].fillna(0)
    project_grade_05 = (earned_points_autograded_05 + earned_points_free_response_05) / (
        max_points_autograded_05 + max_points_free_response_05
    )

    # Total
    project_total_points = (project_grade_01 + project_grade_02 + project_grade_03 + project_grade_04 + project_grade_05) / 5

    return project_total_points


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def lateness_penalty(col):
    # Address lateness column value and converts it datetime variable (hours)
    lateness_hours = pd.to_timedelta(col).dt.total_seconds() / 3600

    # Define border (in hours)
    grace_period = 2
    week = 24 * 7
    two_weeks = 24 * 14

    # Build conditions as boolean
    conditions = [
        (lateness_hours <= grace_period),
        (lateness_hours <= week + grace_period),
        (lateness_hours <= two_weeks + grace_period),
        (lateness_hours >  two_weeks + grace_period),
    ]
    # Assigned values to the conditions
    values = [1.0, 0.9, 0.7, 0.4]

    # Combine conditions with values to np array
    out = np.select(conditions, values)

    # Transform to series object
    return pd.Series(out, index=col.index, dtype=float)

# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def process_labs(grades):
    # lab01
    lab01_earned_points = grades["lab01"]
    lab01_max_points = grades["lab01 - Max Points"] 
    penalty_lab01 = lateness_penalty(grades["lab01 - Lateness (H:M:S)"])
    lab01_total = lab01_earned_points / lab01_max_points * penalty_lab01

    # lab02
    lab02_earned_points = grades["lab02"]
    lab02_max_points = grades["lab02 - Max Points"] 
    penalty_lab02 = lateness_penalty(grades["lab02 - Lateness (H:M:S)"])
    lab02_total = lab02_earned_points / lab02_max_points * penalty_lab02

    # lab03
    lab03_earned_points = grades["lab03"]
    lab03_max_points = grades["lab03 - Max Points"] 
    penalty_lab03 = lateness_penalty(grades["lab03 - Lateness (H:M:S)"])
    lab03_total = lab03_earned_points / lab03_max_points * penalty_lab03

    # lab04
    lab04_earned_points = grades["lab04"]
    lab04_max_points = grades["lab04 - Max Points"] 
    penalty_lab04 = lateness_penalty(grades["lab04 - Lateness (H:M:S)"])
    lab04_total = lab04_earned_points / lab04_max_points * penalty_lab04

    # lab05
    lab05_earned_points = grades["lab05"]
    lab05_max_points = grades["lab05 - Max Points"] 
    penalty_lab05 = lateness_penalty(grades["lab05 - Lateness (H:M:S)"])
    lab05_total = lab05_earned_points / lab05_max_points * penalty_lab05

    # lab06
    lab06_earned_points = grades["lab06"]
    lab06_max_points = grades["lab06 - Max Points"] 
    penalty_lab06 = lateness_penalty(grades["lab06 - Lateness (H:M:S)"])
    lab06_total = lab06_earned_points / lab06_max_points * penalty_lab06

    # lab07
    lab07_earned_points = grades["lab07"]
    lab07_max_points = grades["lab07 - Max Points"] 
    penalty_lab07 = lateness_penalty(grades["lab07 - Lateness (H:M:S)"])
    lab07_total = lab07_earned_points / lab07_max_points * penalty_lab07

    # lab08
    lab08_earned_points = grades["lab08"]
    lab08_max_points = grades["lab08 - Max Points"] 
    penalty_lab08 = lateness_penalty(grades["lab08 - Lateness (H:M:S)"])
    lab08_total = lab08_earned_points / lab08_max_points * penalty_lab08

    # lab09
    lab09_earned_points = grades["lab09"]
    lab09_max_points = grades["lab09 - Max Points"] 
    penalty_lab09 = lateness_penalty(grades["lab09 - Lateness (H:M:S)"])
    lab09_total = lab09_earned_points / lab09_max_points * penalty_lab09

    # Return as DataFrame, index initilized based on grades df
    return pd.DataFrame({
        "lab01": lab01_total,
        "lab02": lab02_total,
        "lab03": lab03_total,
        "lab04": lab04_total,
        "lab05": lab05_total,
        "lab06": lab06_total,
        "lab07": lab07_total,
        "lab08": lab08_total,
        "lab09": lab09_total,
    }, index=grades.index)



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def lab_total(processed):
    # Initialize list
    averages = []
    # For loop which extracts the lowest value from list and builds average
    for i in range(len(processed)):
        row = processed.iloc[i]
        row_list = list(row.values)
        min_val = min(row_list)
        row_list.remove(min_val)
        average = sum(row_list) / len(row_list)
        averages.append(average)
    # Transform to Series object 
    return pd.Series(averages, index=processed.index)


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def total_points(grades):
    ...


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def final_grades(total):
    ...

def letter_proportions(total):
    ...


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def raw_redemption(final_breakdown, question_numbers):
    ...
    
def combine_grades(grades, raw_redemption_scores):
    ...


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def z_score(ser):
    ...
    
def add_post_redemption(grades_combined):
    ...


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def total_points_post_redemption(grades_combined):
    ...
        
def proportion_improved(grades_combined):
    ...


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def section_most_improved(grades_analysis):
    ...
    
def top_sections(grades_analysis, t, n):
    ...


# ---------------------------------------------------------------------
# QUESTION 12
# ---------------------------------------------------------------------


def rank_by_section(grades_analysis):
    ...







# ---------------------------------------------------------------------
# QUESTION 13
# ---------------------------------------------------------------------


def letter_grade_heat_map(grades_analysis):
    ...
