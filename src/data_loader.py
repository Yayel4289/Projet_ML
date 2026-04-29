from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COMMUNE_ID_COL = "CommuneID"
COMMUNE_NAME_COL = "CommuneName"
CANTON_ID_COL = "CantonID"
CANTON_NAME_COL = "CantonName"
COMMUNE_COLS = (COMMUNE_ID_COL, COMMUNE_NAME_COL)
CANTON_COLS = (CANTON_ID_COL, CANTON_NAME_COL)
CANTON_COMMUNE_COLS = CANTON_COLS + COMMUNE_COLS

def rename_cols(df: pd.DataFrame, col_names: Tuple[str], hasBlank=False) -> pd.DataFrame:
    return df.rename(columns={df.columns[i]: col_names[i] for i in range(len(col_names))})

def remove_first_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.iloc[n:].reset_index(drop=True)

def remove_last_blanks(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[:df[COMMUNE_ID_COL].isna().idxmax()] # takes data until blank

def format_commune_cols(df: pd.DataFrame) -> pd.DataFrame:
    df[[COMMUNE_ID_COL]] = df[[COMMUNE_ID_COL]].astype(np.int64)
    df[[COMMUNE_NAME_COL]] = df[[COMMUNE_NAME_COL]].astype(str)
    return df

def format_canton_commune_cols(df: pd.DataFrame, hasBlank=False) -> pd.DataFrame:
    df[[CANTON_ID_COL, COMMUNE_ID_COL]] = df[[CANTON_ID_COL, COMMUNE_ID_COL]].astype(np.int64)
    df[[CANTON_NAME_COL, COMMUNE_NAME_COL]] = df[[CANTON_NAME_COL, COMMUNE_NAME_COL]].astype(str)
    return df

def merge_dfs(left: pd.DataFrame, right: pd.DataFrame, on: List[str]) -> pd.DataFrame:
    return pd.merge(
        left, 
        right, 
        on=on,
        how="outer",
        sort=False
    )

def format_types(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_str = ["CantonName", "CommuneName", "Einheit"]
    df[columns_to_str] = df[columns_to_str].astype(str)

    columns_to_int = ["CantonID", "CommuneID"]
    df[columns_to_int] = df[columns_to_int].fillna(0).astype(np.int64) # obligated to fillna in order to have int

    for col in df.columns:
        if df[col].dtype != 'object': continue
        converted = pd.to_numeric(df[col], errors='coerce')
        df[col] = converted

    return df

def fillna(df: pd.DataFrame) -> pd.DataFrame:
    for str_col in df.select_dtypes(str):
        df[str_col] = df[str_col].fillna("UNKNOWN")

    df["lat"] = df.groupby(CANTON_ID_COL)["lat"].transform(lambda x: x.fillna(x.mean()))
    df["lon"] = df.groupby(CANTON_ID_COL)["lon"].transform(lambda x: x.fillna(x.mean()))
    for float_col in df.select_dtypes(np.float64): # replacing NaN by mean for all float64 cols (except lat, lon)
        if float_col in ("lat", "lon"): continue
        col_mean = df[float_col].mean()
        df[float_col] = df[float_col].fillna(col_mean)
    return df

def load_geodata(data_dir=DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "swiss_communes_geodata.csv"
    df = pd.read_csv(path)
    df = rename_cols(df, COMMUNE_COLS)
    df = format_commune_cols(df)
    df = df.drop(columns=[COMMUNE_NAME_COL])
    return df


def load_income_data(data_dir=DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "statistik-dbst-np-kennzahlen-mit-2017-fr.xlsx"
    df = pd.read_excel(path, sheet_name="Gemeinden - Communes")
    df = rename_cols(df, CANTON_COMMUNE_COLS)
    df = format_canton_commune_cols(df)
    df = df.drop(columns=[CANTON_ID_COL, CANTON_NAME_COL, COMMUNE_NAME_COL])
    return df


def load_commune_portraits(data_dir=DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "je-e-21.03.01.xlsx"
    df = pd.read_excel(path, sheet_name="Schweiz - Gemeinden", header=5)
    df = remove_first_rows(df, 3)
    df = rename_cols(df, COMMUNE_COLS)
    df = remove_last_blanks(df)
    df = format_commune_cols(df)
    df = df.drop(columns=[COMMUNE_NAME_COL])
    return df


def load_previous_vote_results(data_dir=DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "622.00-result-by-canton-district-and-municipality.xlsx"
    df = pd.read_excel(path, sheet_name="Gemeinden", header=5)
    df = remove_first_rows(df, 1)
    df = rename_cols(df, CANTON_COMMUNE_COLS)
    df = remove_last_blanks(df)
    df = format_canton_commune_cols(df)
    return df


def load_all_data(data_dir=DATA_DIR) -> pd.DataFrame:
    geodata = load_geodata(data_dir)
    income = load_income_data(data_dir)
    commune_portraits = load_commune_portraits(data_dir)
    previous_vote = load_previous_vote_results(data_dir)

    df = merge_dfs(previous_vote, income, (COMMUNE_ID_COL))
    df = merge_dfs(df, commune_portraits, COMMUNE_ID_COL)
    df = merge_dfs(df, geodata, COMMUNE_ID_COL)
    df = format_types(df)
    df = fillna(df)

    return df


if __name__ == "__main__":
    df = load_all_data()

    print(df.info())
    print(df.head())
    print(df.isna().sum())
    print("No NaN:", not len(df[df.isna().any(axis=1)]))
