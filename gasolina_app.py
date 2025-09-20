import numpy as np
import streamlit as st
import pandas as pd
import joblib

# ================================
# Cargar el modelo entrenado
# ================================
modelo = joblib.load("modelo_gasolina.pkl")
columnas = joblib.load("columnas_modelo.pkl")

# ================================
# Título y descripción
# ================================
st.title("⛽ Predicción del Precio de la Gasolina en México")
st.markdown("""
Esta aplicación permite predecir el **precio regular de la gasolina** 
según la **entidad**, el **año** y el **mes** seleccionado.
""")

# ================================
# Entradas del usuario
# ================================
entidades = [
    "Ciudad de Mexico", "Aguascalientes", "Baja California", "Baja California Sur",
    "Campeche", "Chiapas", "Chihuahua", "Coahuila", "Colima", "Durango", "Guanajuato",
    "Guerrero", "Hidalgo", "Jalisco", "Mexico", "Michoacan", "Morelos", "Nayarit",
    "Nuevo Leon", "Oaxaca", "Puebla", "Queretaro", "Quintana Roo", "San Luis Potosi",
    "Sinaloa", "Sonora", "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatan", "Zacatecas"
]

entidad = st.selectbox("Selecciona la entidad:", entidades)
anio = st.number_input("Año:", min_value=2017, max_value=2025, value=2023, step=1)
mes = st.slider("Mes:", min_value=1, max_value=12, value=1)

# ================================
# Procesar entrada del usuario
# ================================
df_input = pd.DataFrame({
    "entidad": [entidad],
    "anio": [anio],
    "mes": [mes]
})

df_input = pd.get_dummies(df_input, columns=["entidad"], drop_first=True)
df_input = df_input.reindex(columns=columnas, fill_value=0)

# ================================
# Hacer predicción
# ================================
if st.button("Predecir Precio"):
    prediccion = modelo.predict(df_input)[0]
    st.success(f"💰 El precio estimado de la gasolina es: **${prediccion:.2f} pesos/litro**")
