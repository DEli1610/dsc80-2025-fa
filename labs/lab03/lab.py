import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------

def read_linkedin_survey(dirname):
    target_cols = ["first name", "last name", "current company", "job title", "email", "university"]

    def _clean_cols(df):
        df = df.copy()

        def _norm(c):
            c = str(c).strip().lower()
            c = c.replace("-", " ").replace("_", " ")
            return " ".join(c.split())

        # Spalten normalisieren
        df.columns = [_norm(c) for c in df.columns]

        # Mappings aus der Aufgabenstellung/Vorlesung
        rename_map = {
            # first name
            "first name": "first name", "firstname": "first name", "first": "first name",
            # last name
            "last name": "last name", "lastname": "last name", "surname": "last name", "last": "last name",
            # current company
            "current company": "current company", "company": "current company",
            "company current": "current company", "current employer": "current company", "employer": "current company",
            # job title
            "job title": "job title", "title": "job title", "job": "job title", "position": "job title",
            # email
            "email": "email", "e mail": "email", "mail": "email",
            # university
            "university": "university", "school": "university", "alma mater": "university",
            "university attended": "university",
        }
        df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

        # fehlende Zielspalten auffüllen
        for col in target_cols:
            if col not in df.columns:
                df[col] = pd.NA

        return df[target_cols]

    dirpath = Path(dirname)
    files = sorted(
        [p for p in dirpath.iterdir() if p.is_file() and p.suffix.lower() == ".csv" and p.name.lower().startswith("survey")],
        key=lambda p: p.name.lower(),
    )

    frames = []
    for p in files:
        df = pd.read_csv(p)
        frames.append(_clean_cols(df))

    if not frames:
        return pd.DataFrame(columns=target_cols).reset_index(drop=True)

    out = pd.concat(frames, ignore_index=True)
    return out.reset_index(drop=True)


def com_stats(df):
    uni = df["university"].astype("string")
    title = df["job title"].astype("string")

    # Anteil: Ohio + (case-sensitive) "Programmer" im Titel
    ohio_mask = uni.str.contains("Ohio", case=False, na=False)
    programmer_mask = title.str.contains("Programmer", case=True, na=False)
    denom = int(ohio_mask.sum())
    num = int((ohio_mask & programmer_mask).sum())
    proportion = (num / denom) if denom > 0 else 0.0

    # Anzahl eindeutiger Titel, die mit "Engineer" enden
    unique_titles = pd.Series(title.dropna().str.strip().unique(), dtype="string")
    n_end_engineer = int(unique_titles.apply(lambda t: str(t).endswith("Engineer")).sum())

    # längster Titel (nach Trimmen)
    cleaned = [str(t).strip() for t in title.dropna()]
    longest = max(cleaned, key=len) if cleaned else ""

    # wie oft kommt "manager" (wortgrenzenunabhängig) vor
    n_manager = int(title.str.contains(r"\bmanager\b", case=False, na=False, regex=True).sum())

    return [float(proportion), n_end_engineer, longest, n_manager]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------

def read_student_surveys(dirname):
    dirpath = Path(dirname)
    # nur favorite*.csv, deterministisch sortiert
    files = sorted(
        [p for p in dirpath.iterdir()
         if p.is_file() and p.suffix.lower() == ".csv" and p.name.lower().startswith("favorite")],
        key=lambda p: p.name.lower()
    )

    # Zielindex 1..1000
    out = pd.DataFrame(index=pd.Index(range(1, 1001), name="id"))

    for p in files:
        df = pd.read_csv(p)

        # Wir erwarten exakt eine 'id'-Spalte + 1 weitere Spalte
        # (z. B. name/movie/genre/animal). Falls mehr, nehmen wir die erste Nicht-'id'.
        cols = [c.strip() for c in df.columns]
        df.columns = cols  # keine weitere Normalisierung nötig bei deinen Daten

        if "id" not in df.columns:
            # Wenn wirklich keine 'id' existiert, Datei überspringen (robust)
            continue

        # zweite Spalte bestimmen (erste Spalte, die nicht 'id' ist)
        other_cols = [c for c in df.columns if c != "id"]
        if len(other_cols) == 0:
            # nichts zu joinen
            continue
        resp_col = other_cols[0]  # z. B. "name", "movie", "genre", "animal"

        tmp = df[["id", resp_col]].copy()

        # IDs bereinigen/auf int bringen
        tmp["id"] = pd.to_numeric(tmp["id"], errors="coerce").astype("Int64")
        tmp = tmp.dropna(subset=["id"]).astype({"id": "int"}).set_index("id")

        # join per id; Spaltenname unverändert lassen (wichtig für 'genre' -> check_credit)
        out = out.join(tmp, how="left")

    # sicherstellen, dass Index 1..1000 vollständig ist
    out = out.reindex(range(1, 1001))
    return out


def check_credit(df):
    out = pd.DataFrame(index=df.index)
    out["name"] = df["name"] if "name" in df.columns else pd.NA

    question_cols = [c for c in df.columns if c != "name"]
    num_questions = len(question_cols)

    valid_matrix = pd.DataFrame(index=df.index)

    for c in question_cols:
        s = df[c]
        sc = s.astype("string")
        valid = sc.notna() & sc.str.strip().ne("")
        # Sonderfall „genre“: "(no genres listed)" zählt nicht als Antwort
        if "genre" in c.lower():
            valid = valid & sc.str.strip().str.lower().ne("(no genres listed)")
        valid_matrix[c] = valid

    if num_questions > 0:
        threshold = int(np.ceil(num_questions * 0.5))
        individual5 = (valid_matrix.sum(axis=1) >= threshold).astype(int) * 5
    else:
        individual5 = pd.Series(0, index=df.index)

    if num_questions > 0:
        answered_counts = valid_matrix.sum(axis=0)  # wie viele haben je Frage geantwortet?
        classwide_hits = (answered_counts >= 0.9 * 1000).sum()  # mind. 90% der 1000 IDs
        classwide_bonus = min(2, int(classwide_hits))
    else:
        classwide_bonus = 0

    out["ec"] = individual5 + classwide_bonus
    return out


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------

def most_popular_procedure(pets, procedure_history):
    if procedure_history.empty:
        return ""
    counts = (
        procedure_history.groupby("ProcedureType")
        .size()
        .rename("count")
        .reset_index()
    )
    top = counts.sort_values(["count", "ProcedureType"], ascending=[False, True]).iloc[0]["ProcedureType"]
    return str(top)


def pet_name_by_owner(owners, pets):
    pet_lists = (
        pets.assign(_pet=pets["Name"].astype(str).str.strip())
            .dropna(subset=["OwnerID"])
            .groupby("OwnerID")["_pet"]
            .apply(lambda s: ", ".join(sorted(s.tolist())))
            .rename("pet_name")
    )

    merged = owners.merge(pet_lists, on="OwnerID", how="left")
    merged["pet_name"] = merged["pet_name"].fillna("")

    out = merged.set_index("Name")["pet_name"]
    out.index.name = None
    return out


def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    hist = procedure_history.merge(
        procedure_detail[["ProcedureType", "ProcedureSubCode", "Price"]],
        on=["ProcedureType", "ProcedureSubCode"],
        how="left"
    )
    hist["Price"] = pd.to_numeric(hist["Price"], errors="coerce").fillna(0)

    hp = hist.merge(pets[["PetID", "OwnerID"]], on="PetID", how="left")
    hpo = hp.merge(owners[["OwnerID", "City"]], on="OwnerID", how="left")

    out = hpo.groupby("City")["Price"].sum()
    out.name = "total_cost"
    return out


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

def average_seller(sales):
    out = (
        sales.groupby("Name", as_index=True)["Total"]
             .mean()
             .to_frame("Average Sales")
    )
    return out


def product_name(sales):
    return sales.pivot_table(
        index="Name",
        columns="Product",
        values="Total",
        aggfunc="sum",
    )


def count_product(sales):
    out = sales.pivot_table(
        index=["Product", "Name"],
        columns="Date",
        values="Total",
        aggfunc="count",
        fill_value=0,
    )
    out.columns.name = "Date"
    return out


def total_by_month(sales):
    df = sales.copy()
    # Datumsformat gemäß Beispieldaten: "MM.DD.YYYY"
    df["Month"] = pd.to_datetime(df["Date"], format="%m.%d.%Y").dt.month_name()

    out = df.pivot_table(
        index=["Name", "Product"],
        columns="Month",
        values="Total",
        aggfunc="sum",
        fill_value=0,
    )
    out.columns.name = "Month"
    return out
