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

def format_canton_commune_cols(df: pd.DataFrame) -> pd.DataFrame:
    df[[CANTON_ID_COL, COMMUNE_ID_COL]] = df[[CANTON_ID_COL, COMMUNE_ID_COL]].astype(np.int64)
    df[[CANTON_NAME_COL, COMMUNE_NAME_COL]] = df[[CANTON_NAME_COL, COMMUNE_NAME_COL]].astype(str)
    return df

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

def merge_dfs(left: pd.DataFrame, right: pd.DataFrame, on: List[str]) -> pd.DataFrame:
    return pd.merge(
        left, 
        right, 
        on=on,
        how="outer",
        sort=False
    )

def get_raws(train_df: pd.DataFrame, test_df: pd.DataFrame):
    leakage_columns = [
        'eingelegte Stimmzettel', 'Stimmbeteiligung', 'leere Stimmzettel', 
        'ungültige Stimmzettel', 'gültige Stimmen', 'Ja-Stimmen', 'Nein-Stimmen', 'Ja in Prozent'
    ]
    # only selecting numbers (others are irrelevant) and removing vote results (data leakage)
    X_train_raw = train_df.select_dtypes(include=[np.number]).drop(columns=[c for c in leakage_columns if c in train_df.columns])
    X_test_raw = test_df[X_train_raw.columns]
    return X_train_raw, X_test_raw

def fillna(df: pd.DataFrame) -> pd.DataFrame: # alternative to SimpleImputer with mean strat, only using this because of the position NaNs
    # mean by canton, more accurate
    df["lat"] = df.groupby(CANTON_ID_COL)["lat"].transform(lambda x: x.fillna(x.mean()))
    df["lon"] = df.groupby(CANTON_ID_COL)["lon"].transform(lambda x: x.fillna(x.mean()))
    for float_col in df.select_dtypes(np.float64): # replacing NaN by mean for all float64 cols (except lat, lon) (= mean SimpleImputer)
        if float_col in ("lat", "lon"): continue
        col_mean = df[float_col].mean()
        df[float_col] = df[float_col].fillna(col_mean)
    return df

def load_results_train(data_dir=DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "results_train.csv"
    df = pd.read_csv(path)
    df = rename_cols(df, CANTON_COMMUNE_COLS)
    df = format_canton_commune_cols(df)
    return df

def load_results_test(data_dir=DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "results_test.csv"
    df = pd.read_csv(path)
    df = rename_cols(df, CANTON_COMMUNE_COLS)
    df = format_canton_commune_cols(df)
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


def load_previous_vote_results(data_dir=DATA_DIR, keep_canton_commune_cols=False) -> pd.DataFrame:
    path = Path(data_dir) / "622.00-result-by-canton-district-and-municipality.xlsx"
    df = pd.read_excel(path, sheet_name="Gemeinden", header=5)
    df = df.add_suffix('_622') # similar column names with results_train
    df = remove_first_rows(df, 1)
    df = rename_cols(df, CANTON_COMMUNE_COLS)
    df = remove_last_blanks(df)
    df = format_canton_commune_cols(df)
    if not keep_canton_commune_cols:
        df = df.drop(columns=[CANTON_ID_COL, CANTON_NAME_COL, COMMUNE_NAME_COL])
    return df

def load_all_data(data_dir=DATA_DIR) -> pd.DataFrame:
    geodata = load_geodata(data_dir)
    income = load_income_data(data_dir)
    commune_portraits = load_commune_portraits(data_dir)
    previous_vote = load_previous_vote_results(data_dir, True)

    df = merge_dfs(previous_vote, income, (COMMUNE_ID_COL))
    df = merge_dfs(df, commune_portraits, COMMUNE_ID_COL)
    df = merge_dfs(df, geodata, COMMUNE_ID_COL)
    df = format_types(df)
    df = fillna(df)

    return df

def get_train_test_dfs(data_dir=DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = load_results_train(data_dir)
    test_df = load_results_test(data_dir)
    geodata = load_geodata(data_dir)
    income = load_income_data(data_dir)
    commune_portraits = load_commune_portraits(data_dir)
    previous_vote = load_previous_vote_results(data_dir)

    for dataset in (previous_vote, income, commune_portraits, geodata):
        train_df = train_df.merge(dataset, on=COMMUNE_ID_COL, how="left")
        test_df = test_df.merge(dataset, on=COMMUNE_ID_COL, how="left")
    train_df = format_types(train_df)
    test_df = format_types(test_df)

    y_train = train_df["Ja in Prozent"]
    X_train_raw, X_test_raw = get_raws(train_df, test_df)
    X_train, X_test= fillna(X_train_raw), fillna(X_test_raw)

    return X_train, X_test, y_train


if __name__ == "__main__":
    df = load_all_data()

    print(df.info())
    print(df.head())
    print(df.isna().sum())
    print("No NaN:", not len(df[df.isna().any(axis=1)]))

    X_train, X_test, y_train = get_train_test_dfs()
    X_train.info()
    print(X_train.head(5))
    X_test.info()
    print(X_test.head(5))
    y_train.info()
    print(y_train.head(5))
