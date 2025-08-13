import os
import sys


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def load_data(file_path: str):
    df = pd.read_csv(file_path, header=0, index_col=0)
    return df


def plot_data(df: pd.DataFrame, target_col: str):
    plt.figure(figsize=(10, 5))
    df[target_col].plot()
    plt.show()

def preprocess_data(df: pd.DataFrame):
    pass


def main():
    print("Hello from data-with-darts!")
    df = load_data("DATA/train.csv")
    print(df.head())
    plot_data(df, "전력소비량(kWh)")

    preprocess_data(df)




if __name__ == "__main__":
    main()