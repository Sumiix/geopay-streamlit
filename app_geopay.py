import streamlit as st
import pandas as pd
import joblib

# Carregar model i columnes utilitzades durant l'entrenament
model = joblib.load("model_geopay.pkl")
columnes_model = joblib.load("columnes_model.pkl")

st.title("GeoPay - Detecció de frau")

st.write(
    "Introdueix les dades d'una operació bancària per estimar si pot ser sospitosa."
)

# Camps d'entrada
amount = st.number_input("Import de l'operació", min_value=0.0, value=1000.0)
oldbalanceOrig = st.number_input("Saldo origen abans de l'operació", min_value=0.0, value=1000.0)
newbalanceOrig = st.number_input("Saldo origen després de l'operació", min_value=0.0, value=0.0)
oldbalanceDest = st.number_input("Saldo destí abans de l'operació", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("Saldo destí després de l'operació", min_value=0.0, value=1000.0)

tipus_operacio = st.selectbox(
    "Tipus d'operació",
    ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)

if st.button("Analitzar operació"):

    # Variables derivades
    balanceOrig_diff = oldbalanceOrig - newbalanceOrig
    balanceDest_diff = newbalanceDest - oldbalanceDest
    high_amount = 1 if amount > 100000 else 0

    # Crear registre de l'operació
    operacio = {
        "amount": amount,
        "oldbalanceOrig": oldbalanceOrig,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "balanceOrig_diff": balanceOrig_diff,
        "balanceDest_diff": balanceDest_diff,
        "high_amount": high_amount,
        "type_CASH_OUT": 1 if tipus_operacio == "CASH_OUT" else 0,
        "type_DEBIT": 1 if tipus_operacio == "DEBIT" else 0,
        "type_PAYMENT": 1 if tipus_operacio == "PAYMENT" else 0,
        "type_TRANSFER": 1 if tipus_operacio == "TRANSFER" else 0
    }

    # Convertir a DataFrame
    entrada = pd.DataFrame([operacio])

    # Assegurar que les columnes coincideixen amb les del model
    entrada = entrada.reindex(columns=columnes_model, fill_value=0)

    # Predicció
    prediccio = model.predict(entrada)[0]
    probabilitat = model.predict_proba(entrada)[0][1]

    # Mostrar operació introduïda
    st.subheader("Operació introduïda")
    st.dataframe(entrada)

    # Mostrar resultat
    st.subheader("Resultat de GeoPay")

    if prediccio == 1:
        st.error("Operació sospitosa: possible frau")
    else:
        st.success("Operació aparentment normal")

    st.write("Probabilitat estimada de frau:", round(probabilitat, 4))

    st.write("Diferència saldo origen:", balanceOrig_diff)
    st.write("Diferència saldo destí:", balanceDest_diff)