"""Train and save two models on the coffee analysis data.

model_1.pickle: LinearRegression predicting rating from 100g_USD.
model_2.pickle: DecisionTreeRegressor predicting rating from 100g_USD
                and the numerically encoded roast (roast_cat).
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

DATA_URL = (
    "https://raw.githubusercontent.com/leontoddjohnson/datasets/"
    "refs/heads/main/data/coffee_analysis.csv"
)

# Ordered from lightest to darkest so the numbers carry meaning.
ROAST_MAP = {
    "Light": 0,
    "Medium-Light": 1,
    "Medium": 2,
    "Medium-Dark": 3,
    "Dark": 4,
}


def roast_category(roast):
    """Map a roast label to a number; missing or unknown roasts give NaN."""
    return ROAST_MAP.get(roast, np.nan)


def main():
    df = pd.read_csv(DATA_URL)
    y = df["rating"].values

    # Exercise 1: linear regression on 100g_USD only.
    model_1 = LinearRegression()
    model_1.fit(df[["100g_USD"]].values, y)

    with open("model_1.pickle", "wb") as f:
        pickle.dump(model_1, f)

    # Exercise 2: decision tree on 100g_USD and the encoded roast.
    # Missing roasts stay NaN, which DecisionTreeRegressor supports.
    df["roast_cat"] = df["roast"].map(roast_category)

    model_2 = DecisionTreeRegressor()
    model_2.fit(df[["100g_USD", "roast_cat"]].values, y)

    with open("model_2.pickle", "wb") as f:
        pickle.dump(model_2, f)


if __name__ == "__main__":
    main()
