import streamlit as st
import pandas as pd
import os
import joblib
from train import preprocessing

st.title("Enregistrement de données de qualité de l'eau")

# Champs de saisie
ph = st.number_input("pH", format="%.2f")
hardness = st.number_input("Hardness", format="%.2f")
solids = st.number_input("Solids", format="%.2f")
chloramines = st.number_input("Chloramines", format="%.2f")
sulfate = st.number_input("Sulfate", format="%.2f")
conductivity = st.number_input("Conductivity", format="%.2f")
organic_carbon = st.number_input("Organic Carbon", format="%.2f")
trihalomethanes = st.number_input("Trihalomethanes", format="%.2f")
turbidity = st.number_input("Turbidity", format="%.2f")
model = joblib.load("modele_voting.joblib")

if st.button("Prédire"):
    features = np.array([[entier1, entier2, entier3, entier4, cat1_encoded, cat2_encoded, fl]])
    
# Création du dataframe à partir des inputs
    data = pd.DataFrame({
    'ph': [ph],
    'Hardness': [hardness],
    'Solids': [solids],
    'Chloramines': [chloramines],
    'Sulfate': [sulfate],
    'Conductivity': [conductivity],
    'Organic_carbon': [organic_carbon],
    'Trihalomethanes': [trihalomethanes],
    'Turbidity': [turbidity]
    })

    data = preprocessing(data)
    prediction = model.predict(data)
    st.success(f"Résultat de la prédiction : {prediction[0]}")

