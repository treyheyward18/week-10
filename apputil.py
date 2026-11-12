"""Helpers for loading the saved coffee models and making predictions."""

import pickle

import numpy as np
import pandas as pd

from train import ROAST_MAP, roast_category


def load_model(path):
    """Load a pickled model from the given path."""
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_rating(df_X):
    """Predict coffee ratings from a dataframe of 100g_USD and roast.

    Rows with a known roast use model_2 (price + roast). Rows whose roast
    is missing or not in the training data fall back to model_1 (price).
    """
    model_1 = load_model("model_1.pickle")
    model_2 = load_model("model_2.pickle")

    prices = pd.to_numeric(df_X["100g_USD"], errors="raise").values
    roast_cat = df_X["roast"].map(roast_category).values
    known = df_X["roast"].isin(ROAST_MAP.keys()).values

    y_pred = np.empty(len(df_X), dtype=float)

    if known.any():
        X2 = np.column_stack([prices[known], roast_cat[known]])
        y_pred[known] = model_2.predict(X2)

    if (~known).any():
        X1 = prices[~known].reshape(-1, 1)
        y_pred[~known] = model_1.predict(X1)

    return y_pred
