import streamlit as st
import pandas as pd
import joblib

model=joblib.load("decision_tree_regresion.pkl")
scaler=joblib.load("scal.pkl")
requred_col=joblib.load("colum.pkl")

st.header("Medical Ensurance Prediction Model")

st.markdown("")