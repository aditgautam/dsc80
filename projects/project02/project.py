# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'

from IPython.display import display

# DSC 80 preferred styles
pio.templates["dsc80"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+dsc80"
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def clean_loans(loans: pd.DataFrame):
    cleaned = loans.copy()

    cleaned["issue_d"] = pd.to_datetime(cleaned["issue_d"].astype(str).str.strip(), format="%b-%Y")
    cleaned["term"] = cleaned["term"].astype(str).str.extract(r"(\d+)").astype(int)
    
    emp = cleaned["emp_title"].astype("string").str.lower().str.strip()
    emp = emp.where(emp.ne("rn"), "registered nurse")
    
    cleaned["emp_title"] = emp
    cleaned["term_end"] = cleaned["issue_d"] + cleaned["term"].map(lambda m: pd.DateOffset(months=int(m)))
    return cleaned


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def correlations(df, pairs):
    idx = [f"r_{a}_{b}" for a, b in pairs]
    vals = [pd.to_numeric(df[a], errors="coerce").corr(pd.to_numeric(df[b], errors="coerce"))
            for a, b in pairs]
    return pd.Series(vals, index=idx, dtype="float64")



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_boxplot(loans: pd.DataFrame):
    bins = [580, 670, 740, 800, 850]
    labels = ["[580, 670)", "[670, 740)", "[740, 800)", "[800, 850)"]
    scores = pd.cut(loans["fico_range_low"], bins=bins, right=False, labels=labels)
    df = loans.assign(score_bin=scores.astype(str))

    fig = px.box(
        df,
        x="score_bin",
        y="int_rate",
        color="term",
        labels={
            "score_bin": "Credit Score Range",
            "int_rate": "Interest Rate (%)",
            "term": "Loan Length (Months)"
        },
        title="Interest Rate vs. Credit Score",
        color_discrete_map={36: "purple", 60: "gold"}
    )
    return fig


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def ps_test(loans, N):
    has_ps = loans["desc"].notna()
    x = loans["int_rate"].to_numpy()
    g = has_ps.to_numpy()

    obs = x[g].mean() - x[~g].mean()

    sims = np.empty(N, dtype=float)
    for i in range(N):
        gp = np.random.permutation(g)
        sims[i] = x[gp].mean() - x[~gp].mean()

    return float((sims >= obs).mean())
    
def missingness_mechanism():
    return 2
    
def argument_for_nmar():
    return "" \
    "One who is applying for a loan in a unique financial" \
    "scenario might be more likely to include a personal statement" \
    "not necessarily taking into account the rate"


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def tax_owed(income, brackets):
    brackets = sorted(brackets, key=lambda x: x[1])
    tax = 0.0
    for i, (rate, lower) in enumerate(brackets):
        upper = brackets[i+1][1] if i+1 < len(brackets) else income
        if income > lower:
            tax += (min(income, upper) - lower) * rate
        else:
            break
    return float(tax)


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_state_taxes(state_taxes_raw: pd.DataFrame): 
    df = state_taxes_raw.copy()

    df = df.dropna(how="all")

    df["State"] = df["State"].astype("string").str.strip()
    df.loc[df["State"].str.match(r"^\(.*\)$", na=False), "State"] = pd.NA
    df["State"] = df["State"].ffill()

    rate_str = df["Rate"].astype("string").str.strip().str.lower()
    rate_num = pd.to_numeric(rate_str.str.extract(r"([\d.]+)")[0], errors="coerce")/100.0
    mask_none = rate_str.eq("none")
    df["Rate"] = rate_num.where(~mask_none, 0.0).round(2)

    ll_str = df["Lower Limit"].astype("string").str.replace(r"[,$\s]", "", regex=True)
    df["Lower Limit"] = pd.to_numeric(ll_str, errors="coerce").fillna(0).astype(int)

    df.loc[df["Rate"].eq(0.0), "Lower Limit"] = 0

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def state_brackets(state_taxes: pd.DataFrame):
    df = state_taxes.sort_values(["State", "Lower Limit"])
    out = (df.groupby("State")[["Rate", "Lower Limit"]]
            .apply(lambda g: list(map(tuple, g.to_records(index=False))))
            .to_frame(name="bracket_list"))
    return out
    
def combine_loans_and_state_taxes(loans: pd.DataFrame, state_taxes: pd.DataFrame):
    # Start by loading in the JSON file.
    # state_mapping is a dictionary; use it!
    import json
    state_mapping_path = Path('data') / 'state_mapping.json'
    with open(state_mapping_path, 'r') as f:
        state_mapping = json.load(f)
        
    # Now it's your turn:
    sb = state_brackets(state_taxes).reset_index()
    sb["State"] = sb["State"].map(lambda s: state_mapping.get(s, s))

    out = loans.copy().rename(columns={"addr_state": "State"})
    out = out.merge(sb, on="State", how="left")
    return out


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def find_disposable_income(loans_with_state_taxes: pd.DataFrame):
    FEDERAL_BRACKETS = [
     (0.1, 0), 
     (0.12, 11000), 
     (0.22, 44725), 
     (0.24, 95375), 
     (0.32, 182100),
     (0.35, 231251),
     (0.37, 578125)
    ]
    df = loans_with_state_taxes.copy()

    df["federal_tax_owed"] = df["annual_inc"].apply(
        lambda inc: tax_owed(float(inc), FEDERAL_BRACKETS)
    )

    df["state_tax_owed"] = df.apply(
        lambda r: tax_owed(float(r["annual_inc"]), r["brackets_list"])
        if isinstance(r.get("brackets_list", None), list) and len(r["brackets_list"]) > 0
        else 0.0,
        axis=1
    )

    df["disposable_income"] = (
        df["annual_inc"] - df["federal_tax_owed"] - df["state_tax_owed"]
    )
    return df


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def aggregate_and_combine(loans, keywords, quantitative_column, categorical_column):
    out = pd.DataFrame()

    for kw in keywords:
        mask = loans["emp_title"].str.contains(kw, na=False)

        grouped = (loans.loc[mask]
                          .groupby(categorical_column)[quantitative_column]
                          .mean())

        overall = loans.loc[mask, quantitative_column].mean()

        s = pd.concat([grouped, pd.Series({"Overall": overall})])
        out[f"{kw}_mean_{quantitative_column}"] = s

    return out


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def exists_paradox(loans, keywords, quantitative_column, categorical_column):
    #what is the clean solution?
    tbl = aggregate_and_combine(loans, keywords, quantitative_column, categorical_column)
    return bool(
        (tbl.iloc[:-1][f"{keywords[0]}_mean_{quantitative_column}"] >
         tbl.iloc[:-1][f"{keywords[1]}_mean_{quantitative_column}"]).all()
        and
        tbl.iloc[-1][f"{keywords[0]}_mean_{quantitative_column}"] <
        tbl.iloc[-1][f"{keywords[1]}_mean_{quantitative_column}"]
    )
    
def paradox_example(loans):
    #pairs to try
    kws = [
        ("engineer", "teacher"),
        ("manager", "nurse"),
        ("analyst", "nurse"),
        ("mechanic", "nurse"),
        ("sales", "nurse"),
        ("driver", "nurse"),
        ("teacher", "nurse"),
        ("engineer", "manager"),
    ]
    quants = ["loan_amnt", "int_rate"]
    cats = ["verification_status", "grade", "home_ownership", "term", "purpose"]

    #categories to not iterate over
    banned = ("engineer", "nurse", "loan_amnt", "home_ownership")

    for k1, k2 in kws:
        for q in quants:
            for c in cats:
                if (k1, k2, q, c) == banned:
                    continue
                try:
                    if exists_paradox(loans, [k1, k2], q, c):
                        return {
                            "loans": loans,
                            "keywords": [k1, k2],
                            "quantitative_column": q,
                            "categorical_column": c,
                        }
                except Exception:
                    #continue if loop breaks
                    pass

    
    return {
        "loans": loans,
        "keywords": ["engineer", "nurse"],
        "quantitative_column": "loan_amnt",
        "categorical_column": "verification_status",
    }
