"""
This script cleans census.csv

Author: BaoNDQ
Creation Date: Aug 18 2024
"""

import pandas as pd
import numpy as np


def clean_census_data(out_path:str="data/cleaned_census.csv"):
    df = pd.read_csv("data/census.csv", skipinitialspace=True)
    df = df.replace(to_replace="?", value=np.nan)
    df = df.dropna()
    df.to_csv(out_path, index=False)


if __name__ == "__main__":

    clean_census_data(out_path="data/cleaned_census.csv")
