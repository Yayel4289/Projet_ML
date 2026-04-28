from pathlib import Path
from typing import List, Tuple
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COMMUNE_ID_COL = "CommuneID"
COMMUNE_NAME_COL = "CommuneName"
CANTON_ID_COL = "CantonID"
CANTON_NAME_COL = "CantonName"
COMMUNE_COLS = (COMMUNE_ID_COL, COMMUNE_NAME_COL)
CANTON_COLS = (CANTON_ID_COL, CANTON_NAME_COL)
CANTON_COMMUNE_COLS = CANTON_COLS + COMMUNE_COLS

def rename_cols(df: pd.DataFrame, col_names: Tuple[str]) -> None:
    df.rename(columns={df.columns[i]: col_names[i] for i in range(len(col_names))}, inplace=True)

def rename_commune_cols(df: pd.DataFrame) -> None:
    rename_cols(df, COMMUNE_COLS)

def rename_canton_commune_cols(df: pd.DataFrame) -> None:
    rename_cols(df, CANTON_COMMUNE_COLS)

def merge_dfs(left: pd.DataFrame, right: pd.DataFrame, on: List[str]) -> pd.DataFrame:
    return pd.merge(
        left, 
        right, 
        on=on,
        how="outer",
        sort=False
    )

def load_geodata(data_dir=DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "swiss_communes_geodata.csv"
    df = pd.read_csv(path)
    rename_commune_cols(df)
    return df


def load_income_data(data_dir=DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "statistik-dbst-np-kennzahlen-mit-2017-fr.xlsx"
    df = pd.read_excel(path, sheet_name="Gemeinden - Communes")
    rename_canton_commune_cols(df)
    return df


def load_commune_portraits(data_dir=DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "je-e-21.03.01.xlsx"
    df = pd.read_excel(path, sheet_name="Schweiz - Gemeinden", header=5)
    rename_commune_cols(df)
    return df


def load_previous_vote_results(data_dir=DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "622.00-result-by-canton-district-and-municipality.xlsx"
    df = pd.read_excel(path, sheet_name="Gemeinden", header=5)
    rename_canton_commune_cols(df)
    return df


def load_all_data(data_dir=DATA_DIR) -> pd.DataFrame:
    geodata = load_geodata(data_dir)
    income = load_income_data(data_dir)
    commune_portraits = load_commune_portraits(data_dir)
    previous_vote = load_previous_vote_results(data_dir)
    print(geodata.info(), income.info(), commune_portraits.info(), previous_vote.info())

    geodata_income = merge_dfs(geodata, income, COMMUNE_COLS)
    geodata_income_portraits = merge_dfs(geodata_income, commune_portraits, CANTON_COLS)
    df = merge_dfs(geodata_income_portraits, previous_vote, CANTON_COMMUNE_COLS)

    return df


if __name__ == "__main__":
    df = load_all_data()

    print(df.info())
    print(df.head())
