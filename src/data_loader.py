from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_geodata(data_dir=DATA_DIR):
    path = Path(data_dir) / "swiss_communes_geodata.csv"
    df = pd.read_csv(path)
    return df


def load_income_data(data_dir=DATA_DIR):
    path = Path(data_dir) / "statistik-dbst-np-kennzahlen-mit-2017-fr.xlsx"
    df = pd.read_excel(path, sheet_name="Gemeinden - Communes")
    return df


def load_commune_portraits(data_dir=DATA_DIR):
    path = Path(data_dir) / "je-e-21.03.01.xlsx"
    df = pd.read_excel(path, sheet_name="Schweiz - Gemeinden", header=5)
    return df


def load_previous_vote_results(data_dir=DATA_DIR):
    path = Path(data_dir) / "622.00-result-by-canton-district-and-municipality.xlsx"
    df = pd.read_excel(path, sheet_name="Gemeinden", header=5)
    return df


def load_all_data(data_dir=DATA_DIR):
    geodata = load_geodata(data_dir)
    income = load_income_data(data_dir)
    commune_portraits = load_commune_portraits(data_dir)
    previous_vote = load_previous_vote_results(data_dir)

    return geodata, income, commune_portraits, previous_vote


if __name__ == "__main__":
    geodata, income, commune_portraits, previous_vote = load_all_data()

    print("Geodata:", geodata.shape)
    print("Income:", income.shape)
    print("Commune portraits:", commune_portraits.shape)
    print("Previous vote:", previous_vote.shape)