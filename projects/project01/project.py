# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades: pd.DataFrame):
    base = pd.Index(grades.columns).str.split(' - ', n=1).str[0]
    uniq = pd.Series(base.unique())

    def last2_isdigits(s: pd.Series) -> pd.Series:
        return s.str[-2:].str.isnumeric()

    labs = uniq[uniq.str.startswith('lab') &
                (uniq.str.len() == 5) &
                last2_isdigits(uniq)].tolist()

    discs = uniq[uniq.str.startswith('discussion') &
                 (uniq.str.len() == len('discussion') + 2) &
                 last2_isdigits(uniq)].tolist()

    checkpoints = uniq[uniq.str.startswith('project') &
                       (uniq.str.find('_checkpoint') >= 0)].tolist()

    projects = uniq[uniq.str.startswith('project') &
                    (uniq.str.find('_checkpoint') == -1)].tolist()

    midterm = uniq[uniq == 'Midterm'].tolist()
    final   = uniq[uniq == 'Final'].tolist()

    return {
        'lab':        sorted(labs),
        'project':    sorted(projects),
        'midterm':    sorted(midterm),
        'final':      sorted(final),
        'disc':       sorted(discs),
        'checkpoint': sorted(checkpoints),
    }



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades: pd.DataFrame) -> pd.Series:
    names = get_assignment_names(grades)

    proj_cols = [c for c in names['project']
                 if 'checkpoint' not in c.lower() and ' - ' not in c]

    by_project = {}
    for c in proj_cols:
        base = c.split('_')[0]  # 'project02'
        by_project.setdefault(base, []).append(c)

    per_project_prop = []
    for base, parts in by_project.items():
        earned_total = pd.Series(0.0, index=grades.index)
        max_total    = pd.Series(0.0, index=grades.index)
        for part in parts:
            if f"{part} - Max Points" not in grades.columns:
                continue
            earned = pd.to_numeric(grades[part], errors='coerce').fillna(0)
            mx     = pd.to_numeric(grades[f"{part} - Max Points"], errors='coerce')
            earned_total = earned_total.add(earned, fill_value=0)
            max_total    = max_total.add(mx,     fill_value=0)

        proj_prop = (earned_total / max_total).replace([np.inf, -np.inf], np.nan).fillna(0)
        per_project_prop.append(proj_prop)

    if not per_project_prop:
        return pd.Series(0.0, index=grades.index, dtype=float)

    return pd.concat(per_project_prop, axis=1).mean(axis=1).astype(float)



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def lateness_penalty(col: pd.Series) -> pd.Series:
    td = pd.to_timedelta(col, errors='coerce')
    hrs = td.dt.total_seconds().div(3600)

    hrs = hrs.fillna(0).clip(lower=0)

    penalty = pd.Series(1.0, index=col.index, dtype=float)

    penalty.loc[(hrs > 2) & (hrs <= 24*7)] = 0.9

    penalty.loc[(hrs > 24*7) & (hrs <= 24*14)] = 0.7

    penalty.loc[hrs > 24*14] = 0.4

    return penalty


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def process_labs(grades: pd.DataFrame):
    names = get_assignment_names(grades)
    lab_cols = [c for c in names['lab'] if ' - ' not in c]

    out = pd.DataFrame(index=grades.index)
    for c in lab_cols:
        lateness_col = f'{c} - Lateness (H:M:S)'
        max_col = f'{c} - Max Points'

        earned = pd.to_numeric(grades[c], errors='coerce').fillna(0)
        mx     = pd.to_numeric(grades[max_col], errors='coerce')

        norm = (earned / mx).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
        pen  = lateness_penalty(grades[lateness_col])
        out[c] = norm * pen

    return out



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def lab_total(processed):
    if processed.empty:
        return pd.Series([], dtype=float)
    lowest_scores_index = processed.idxmin(axis=1)

    df_without_lowest = processed.copy()
    for row_idx, col_label in lowest_scores_index.items():
        df_without_lowest.loc[row_idx, col_label] = pd.NA

    final_lab_grades = df_without_lowest.mean(axis=1)

    return final_lab_grades


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def total_points(grades: pd.DataFrame) -> pd.Series:
    names = get_assignment_names(grades)

    def mean_normalized(cols):
        cols = [c for c in cols if ' - ' not in c]
        parts = []
        for c in cols:
            if f"{c} - Max Points" not in grades.columns:
                continue
            earned = pd.to_numeric(grades[c], errors='coerce').fillna(0)
            mx = pd.to_numeric(grades[f"{c} - Max Points"], errors='coerce')
            part = (earned / mx).replace([np.inf, -np.inf], np.nan).fillna(0)
            parts.append(part)
        if not parts:
            return pd.Series(0.0, index=grades.index)
        return pd.concat(parts, axis=1).mean(axis=1)

    labs_processed = process_labs(grades)
    labs_prop = lab_total(labs_processed).fillna(0)

    projects_prop = projects_total(grades).fillna(0)

    checkpoint_cols = [c for c in names['checkpoint'] if 'checkpoint' in c.lower()]
    checkpoints_prop = mean_normalized(checkpoint_cols)

    disc_cols = names['disc']
    discussions_prop = mean_normalized(disc_cols)

    midterm_col = next(
        (c for c in names['midterm']
        if ' - ' not in c and f"{c} - Max Points" in grades.columns),
        None
    )
    final_col = next(
        (c for c in names['final']
        if ' - ' not in c and f"{c} - Max Points" in grades.columns),
        None
    )

    def norm_one(colname):
        if colname is None or f"{colname} - Max Points" not in grades.columns:
            return pd.Series(0.0, index=grades.index)
        e = pd.to_numeric(grades[colname], errors='coerce').fillna(0)
        m = pd.to_numeric(grades[f"{colname} - Max Points"], errors='coerce')
        return (e / m).replace([np.inf, -np.inf], np.nan).fillna(0)

    midterm_prop = norm_one(midterm_col)
    final_prop   = norm_one(final_col)

    total = (
        0.20 * labs_prop +
        0.30 * projects_prop +
        0.025 * checkpoints_prop +
        0.025 * discussions_prop +
        0.15 * midterm_prop +
        0.30 * final_prop
    )

    return total.clip(0, 1).astype(float)



# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def final_grades(total: pd.Series) -> pd.Series:
    s = pd.to_numeric(total, errors='coerce').fillna(0).clip(0, 1)
    bins   = [-np.inf, 0.6, 0.7, 0.8, 0.9, np.inf]
    labels = ['F', 'D', 'C', 'B', 'A']
    return pd.cut(s, bins=bins, labels=labels, right=False).astype(str)

def letter_proportions(total: pd.Series) -> pd.Series:
    letters = final_grades(total)
    order = ['A', 'B', 'C', 'D', 'F']
    counts = letters.value_counts().reindex(order, fill_value=0)

    n = int(counts.sum())
    if n == 0:
        return pd.Series(0.0, index=['B','C','A','D','F'])

    props = counts.astype(float) / n
    props = props.sort_values(ascending=False)

    if len(props) > 0:
        props.iloc[-1] = 1.0 - props.iloc[:-1].sum()

    return props



# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def raw_redemption(final_breakdown: pd.DataFrame, question_numbers: list[int]):
    q_cols = [final_breakdown.columns[q] for q in question_numbers]

    earned = final_breakdown[q_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    per_q_max = earned.max(axis=0)

    total_possible = per_q_max.sum()

    if total_possible == 0:
        scores = pd.Series(0.0, index=final_breakdown.index)
    else:
        scores = earned.sum(axis=1) / total_possible

    return pd.DataFrame({
        "PID": final_breakdown["PID"],
        "Raw Redemption Score": scores.astype(float)
    })
    
def combine_grades(grades: pd.DataFrame, raw_redemption_scores: pd.DataFrame) -> pd.DataFrame:
    out = grades.merge(
        raw_redemption_scores[["PID", "Raw Redemption Score"]],
        on="PID",
        how="left",
        validate="1:1",
    )
    out["Raw Redemption Score"] = pd.to_numeric(out["Raw Redemption Score"], errors="coerce").fillna(0.0)
    return out


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------



def z_score(ser: pd.Series) -> pd.Series:
    x = pd.to_numeric(ser, errors="coerce")
    mu = x.mean(skipna=True)
    sd = x.std(ddof=0, skipna=True)
    if sd == 0 or np.isnan(sd):
        return x.apply(lambda v: np.nan if pd.isna(v) else 0.0)
    return (x - mu) / sd

def add_post_redemption(grades_combined: pd.DataFrame) -> pd.DataFrame:
    out = grades_combined.copy()

    mid = pd.to_numeric(out['Midterm'], errors='coerce')
    mx  = pd.to_numeric(out['Midterm - Max Points'], errors='coerce')
    pre = (mid / mx).replace([np.inf, -np.inf], np.nan)
    out['Midterm Score Pre-Redemption'] = pre

    rr   = pd.to_numeric(out['Raw Redemption Score'], errors='coerce')
    rr_z = z_score(rr)
    
    pre_for_stats = pre.fillna(0.0)
    mid_z = z_score(pre_for_stats)
    mu    = pre_for_stats.mean()
    sd    = pre_for_stats.std(ddof=0)

    eps = 1e-12
    eligible = pre.notna() & (pre < 1.0 - eps)
    back = rr_z * sd + mu

    improved = eligible & (rr_z > mid_z + eps) & (back > pre + eps)

    post = pre.copy()
    post.loc[improved] = back.loc[improved]

    out['Midterm Score Post-Redemption'] = post.fillna(0.0).clip(lower=0.0, upper=1.0)
    return out

# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def total_points_post_redemption(grades_combined: pd.DataFrame) -> pd.Series:
    df = grades_combined
    if ("Midterm Score Pre-Redemption" not in df.columns
        or "Midterm Score Post-Redemption" not in df.columns):
        df = add_post_redemption(df)

    df_before = df.drop(columns=[
        "Midterm Score Pre-Redemption",
        "Midterm Score Post-Redemption"
    ], errors="ignore")
    before = total_points(df_before)

    pre  = pd.to_numeric(df["Midterm Score Pre-Redemption"],  errors="coerce").fillna(0)
    post = pd.to_numeric(df["Midterm Score Post-Redemption"], errors="coerce").fillna(0)

    after = before - 0.15 * pre + 0.15 * post
    return after.clip(0, 1).astype(float)


def proportion_improved(grades_combined: pd.DataFrame) -> float:
    df = grades_combined
    if ("Midterm Score Pre-Redemption" not in df.columns
        or "Midterm Score Post-Redemption" not in df.columns):
        df = add_post_redemption(df)

    df_before = df.drop(columns=[
        "Midterm Score Pre-Redemption",
        "Midterm Score Post-Redemption"
    ], errors="ignore")

    total_before = total_points(df_before)
    total_after  = total_points_post_redemption(df)

    rank = {'F': 0, 'D': 1, 'C': 2, 'B': 3, 'A': 4}
    before = final_grades(total_before).map(rank)
    after  = final_grades(total_after).map(rank)

    return float((after > before).mean())


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def section_most_improved(grades_analysis: pd.DataFrame):
    rank = {'F':0,'D':1,'C':2,'B':3,'A':4}
    pre  = grades_analysis['Letter Grade Pre-Redemption'].map(rank)
    post = grades_analysis['Letter Grade Post-Redemption'].map(rank)

    improved = post > pre
    props = improved.groupby(grades_analysis['Section']).mean()
    return props.idxmax()
    
def top_sections(grades_analysis, t, n):
    final_scores = pd.to_numeric(grades_analysis['Final'], errors="coerce")
    final_max = pd.to_numeric(grades_analysis['Final - Max Points'], errors="coerce")
    
    passed = final_scores >= (t * final_max)
    counts = passed.groupby(grades_analysis['Section']).sum()
    valid_sections = counts[counts >= n].index
    return np.sort(valid_sections.values)
    


# ---------------------------------------------------------------------
# QUESTION 12
# ---------------------------------------------------------------------


def rank_by_section(grades_analysis: pd.DataFrame) -> pd.DataFrame:
    sections = np.sort(grades_analysis['Section'].unique())

    sizes = grades_analysis.groupby('Section').size()
    n = int(sizes.max())

    col_data = {}
    for sec in sections:
        sub = grades_analysis[grades_analysis['Section'] == sec]
        order = sub.sort_values(
            'Total Points Post-Redemption', ascending=False, kind='mergesort'
        )['PID'].tolist()
        col_data[sec] = order + [''] * (n - len(order))

    out = pd.DataFrame(col_data, index=range(1, n+1))
    out.index.name = 'Section Rank'
    return out


# ---------------------------------------------------------------------
# QUESTION 13
# ---------------------------------------------------------------------


def letter_grade_heat_map(grades_analysis):
    if 'Letter Grade Post-Redemption' in grades_analysis.columns:
        letter_col = 'Letter Grade Post-Redemption'
    else:
        letter_col = next(
            (c for c in grades_analysis.columns if c.lower().startswith('letter grade post')),
            None
        )
        if letter_col is None:
            raise KeyError("Couldnt find the post-redemption letter-grade column")

    grade_order   = ['A', 'B', 'C', 'D', 'F']
    section_order = sorted(grades_analysis['Section'].unique())

    prop = (
        grades_analysis
        .groupby('Section')[letter_col]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reindex(index=section_order, columns=grade_order, fill_value=0.0)
        .T
    )

    fig = px.imshow(
        prop.values,
        x=prop.columns,
        y=prop.index,
        labels={'x': 'Section', 'y': 'Letter Grade Post-Redemption', 'color': 'color'},
        color_continuous_scale='Blues',
        zmin=0.0, zmax=1.0
    )
    fig.update_layout(title='Distribution of Letter Grades by Section')

    fig.update_yaxes(categoryorder='array', categoryarray=grade_order, title='Letter Grade Post-Redemption')
    fig.update_xaxes(categoryorder='array', categoryarray=section_order, title='Section')

    fig.update_layout(
    coloraxis=dict(
        colorscale='Blues',
        cmin=0, cmax=1,
        colorbar=dict(tick0=0, dtick=0.1)
    )
)

    return fig

