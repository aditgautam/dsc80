# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np
import math


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def read_linkedin_survey(dirname) -> pd.DataFrame:
    dirname = Path(dirname)
    req_cols = ['first name', 'last name', 'current company', 'job title', 'email', 'university']
    aliases = {
        'first name': ['first name','first','firstname'],
        'last name': ['last name','last','lastname','surname'],
        'current company': ['current company','company','employer','current employer'],
        'job title': ['job title','title','job','position'],
        'email': ['email','email address','e-mail'],
        'university': ['university','school','college','alma mater']
    }
    files = [p for p in dirname.iterdir() if p.is_file() and p.name.startswith('survey') and p.suffix == '.csv']
    if not files:
        return pd.DataFrame(columns=req_cols)
    parts = []
    for f in files:
        raw = pd.read_csv(f)
        norm_map = {c.strip().lower().replace('_',' '): c for c in raw.columns}
        cols = {}
        for target in req_cols:
            found = next((norm_map[a] for a in aliases[target] if a in norm_map), None)
            cols[target] = raw[found] if found is not None else pd.Series([pd.NA]*len(raw), index=raw.index)
        df = pd.DataFrame(cols)
        df['job title'] = df['job title'].astype(str).str.strip()
        df['university'] = df['university'].astype(str).str.strip()
        parts.append(df)
    out = pd.concat(parts, axis=0, ignore_index=True)
    out.reset_index(drop=True, inplace=True)
    return out

def com_stats(df: pd.DataFrame):
    ohio = df['university'].astype(str).str.contains('Ohio', case=False, na=False)
    programmer = df['job title'].astype(str).str.contains('Programmer', case=True, na=False)
    prop_ohio_programmer = (ohio & programmer).mean() if len(df) else 0.0
    titles = df['job title'].astype(str).str.strip()
    ends_engineer = titles.str.endswith('Engineer', na=False)
    num_distinct_engineer_end = titles[ends_engineer].unique().size
    longest_title = titles.loc[titles.str.len().idxmax()] if len(df) else ""
    manager_count = titles.str.contains(r'\bmanager\b', case=False, regex=True, na=False).sum()
    return [prop_ohio_programmer, num_distinct_engineer_end, longest_title, int(manager_count)]





# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def read_student_surveys(dirname: Path):
    dirname = Path(dirname)
    if not dirname.exists() or not dirname.is_dir():
        raise FileNotFoundError(dirname)

    files = sorted(dirname.glob("favorite*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f).set_index("id")
        dfs.append(df)
    return pd.concat(dfs, axis=1).sort_index()


def check_credit(df: pd.DataFrame):
    name_col = "name"
    qcols = [c for c in df.columns if c != name_col]

    valid = {}
    for c in qcols:
        m = df[c].notna()
        if "genre" in c.lower():
            m = m & (df[c] != "(no genres listed)")
        valid[c] = m
    valid_df = pd.DataFrame(valid, index=df.index)

    n_q = len(qcols)
    thresh = math.ceil(0.5 * n_q)
    base = (valid_df.sum(axis=1) >= thresh).astype(int) * 5

    class_ok = (valid_df.sum(axis=0) / len(valid_df) >= 0.90)
    bonus = min(2, int(class_ok.sum()))

    ec = (base + bonus).clip(upper=7)
    return pd.DataFrame({"name": df[name_col], "ec": ec}).sort_index()


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_popular_procedure(pets: pd.DataFrame, procedure_history: pd.DataFrame):
    valid = procedure_history[procedure_history["PetID"].isin(pets["PetID"])]
    return valid["ProcedureType"].value_counts().idxmax()

def pet_name_by_owner(owners: pd.DataFrame, pets: pd.DataFrame) -> pd.Series:
    ow = owners.rename(columns={"Name": "FirstName"})
    df = pets.merge(ow[["OwnerID", "FirstName"]], on="OwnerID", how="inner")

    per_owner = df.groupby("OwnerID")["Name"].agg(
        lambda x: x.iloc[0] if len(x) == 1 else x.tolist()
    )

    first_names = df.drop_duplicates("OwnerID").set_index("OwnerID")["FirstName"]
    out = per_owner.to_frame("PetNames").join(first_names)
    return out.set_index("FirstName")["PetNames"]

def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    mh = procedure_history.merge(procedure_detail, on="ProcedureType", how="inner")
    mhp = mh.merge(pets[["PetID", "OwnerID"]], on="PetID", how="inner")
    full = mhp.merge(owners[["OwnerID", "City"]], on="OwnerID", how="inner")
    spent = full.groupby("City")["Price"].sum()
    all_cities = owners["City"].dropna().unique()
    return spent.reindex(all_cities, fill_value=0).sort_index()




# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def average_seller(sales: pd.DataFrame):
    out = sales.groupby("Name", as_index=True)["Total"].mean().reset_index()
    out = out.rename(columns={"Total": "Average Sales"}).set_index("Name")
    return out.fillna(0)

def product_name(sales: pd.DataFrame):
    return sales.pivot_table(index="Name", columns="Product", values="Total", aggfunc="sum")


def count_product(sales: pd.DataFrame):
    out = sales.pivot_table(index=["Product", "Name"],columns="Date", values="Total", aggfunc="count")
    return out.fillna(0)

def total_by_month(sales: pd.DataFrame):
    s = sales.copy()
    s["Month"] = pd.to_datetime(s["Date"], format="%m.%d.%Y").dt.month_name()
    out = s.pivot_table(index=["Name", "Product"], columns="Month", values="Total", aggfunc="sum")
    return out.fillna(0)
