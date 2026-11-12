import pandas as pd
import streamlit as st

from apputil import predict_rating
from train import ROAST_MAP

st.write(
    '''
# Week 10: Coffee Rating Predictor

Enter a price per 100g and a roast to predict the coffee's rating.
If the roast is unknown, the prediction uses price only.
'''
)

price = st.number_input(
    "Price per 100g (USD):", value=None, min_value=0.0, step=0.5
)
roast = st.selectbox(
    "Roast:", list(ROAST_MAP.keys()) + ["Unknown / other"]
)

if price is None:
    st.info("Enter a price to get a prediction.")
elif price <= 0:
    st.error("Price must be greater than 0.")
else:
    df_X = pd.DataFrame([[price, roast]], columns=["100g_USD", "roast"])
    try:
        rating = predict_rating(df_X)[0]
        st.write(f"Predicted rating: **{rating:.1f}**")
    except FileNotFoundError:
        st.error("Model files not found. Run `python train.py` first.")
