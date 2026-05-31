import streamlit as st
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# -----------------------------
# Càrrega del model existent
# -----------------------------
model = joblib.load("model_geopay.pkl")
columnes_model = joblib.load("columnes_model.pkl")


# -----------------------------
# Funció de preprocessament
# -----------------------------
def preprocessar_dataset_frau(df):
    df = df.copy()

    # Eliminació de duplicats
    df = df.drop_duplicates()

    # Comprovació de columnes mínimes necessàries
    columnes_necessaries = [
        "type", "amount", "oldbalanceOrig", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "isFraud"
    ]

    for col in columnes_necessaries:
        if col not in df.columns:
            raise ValueError(f"Falta la columna necessària: {col}")

    # Eliminació de registres sense variables clau
    df = df.dropna(subset=["type", "amount", "isFraud"])

    # Creació de variables derivades
    df["balanceOrig_diff"] = df["oldbalanceOrig"] - df["newbalanceOrig"]
    df["balanceDest_diff"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df["high_amount"] = (
        df["amount"] > df["amount"].quantile(0.95)
    ).astype(int)

    # Codificació de la variable categòrica type
    df = pd.get_dummies(df, columns=["type"], drop_first=True)

    # Eliminació d'identificadors
    columnes_eliminar = ["Orig", "Dest", "nameOrig", "nameDest"]
    df = df.drop(
        columns=[col for col in columnes_eliminar if col in df.columns],
        errors="ignore"
    )

    return df


# -----------------------------
# Títol general
# -----------------------------
st.title("GeoPay - Detecció de frau")

st.write(
    "MVP per estimar si una operació bancària pot ser sospitosa mitjançant un model Random Forest."
)


# -----------------------------
# Pestanyes
# -----------------------------
tab_prediccio, tab_entrenament = st.tabs([
    "Predicció",
    "Entrenament"
])


# -----------------------------
# Pestanya 1: Predicció
# -----------------------------
with tab_prediccio:

    st.subheader("Consulta d'una operació")

    st.write(
        "Introdueix les dades d'una operació bancària per estimar si pot ser sospitosa."
    )

    amount = st.number_input("Import de l'operació", min_value=0.0, value=30000.0)
    oldbalanceOrig = st.number_input("Saldo origen abans de l'operació", min_value=0.0, value=30000.0)
    newbalanceOrig = st.number_input("Saldo origen després de l'operació", min_value=0.0, value=0.0)
    oldbalanceDest = st.number_input("Saldo destí abans de l'operació", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("Saldo destí després de l'operació", min_value=0.0, value=30000.0)

    tipus_operacio = st.selectbox(
        "Tipus d'operació",
        ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    )

    if st.button("Analitzar operació"):

        balanceOrig_diff = oldbalanceOrig - newbalanceOrig
        balanceDest_diff = newbalanceDest - oldbalanceDest
        high_amount = 1 if amount > 100000 else 0

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

        entrada = pd.DataFrame([operacio])
        entrada = entrada.reindex(columns=columnes_model, fill_value=0)

        prediccio = model.predict(entrada)[0]
        probabilitat = model.predict_proba(entrada)[0][1]

        st.subheader("Operació introduïda")
        st.dataframe(entrada)

        st.subheader("Resultat de GeoPay")

        if prediccio == 1:
            st.error("Operació sospitosa: possible frau")
        else:
            st.success("Operació aparentment normal")

        st.write("Probabilitat estimada de frau:", f"{probabilitat * 100:.2f}%")
        st.write("Diferència saldo origen:", balanceOrig_diff)
        st.write("Diferència saldo destí:", balanceDest_diff)


# -----------------------------
# Pestanya 2: Entrenament
# -----------------------------
with tab_entrenament:

    st.subheader("Actualització del model")

    st.write(
        "Puja un nou fitxer CSV amb dades de transaccions per entrenar una nova versió del model GeoPay."
    )

    fitxer_csv = st.file_uploader(
        "Selecciona un fitxer CSV",
        type=["csv"]
    )

    if fitxer_csv is not None:

        df_nou = pd.read_csv(fitxer_csv)

        st.write("Dataset carregat correctament:")
        st.write("Dimensions:", df_nou.shape)
        st.dataframe(df_nou.head())

        if st.button("Entrenar nou model"):

            try:
                df_preparat = preprocessar_dataset_frau(df_nou)

                if "isFraud" not in df_preparat.columns:
                    st.error("El dataset ha de contenir la columna objectiu 'isFraud'.")
                else:
                    X = df_preparat.drop(columns=["isFraud"])
                    y = df_preparat["isFraud"]

                    # Adaptar les columnes a l'estructura del model original
                    X = X.reindex(columns=columnes_model, fill_value=0)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.2,
                        random_state=42,
                        stratify=y
                    )

                    nou_model = RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1
                    )

                    nou_model.fit(X_train, y_train)

                    y_pred = nou_model.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)

                    st.success("Nou model entrenat correctament.")

                    st.write("Accuracy:", round(accuracy, 4))
                    st.write("Precision:", round(precision, 4))
                    st.write("Recall:", round(recall, 4))
                    st.write("F1-score:", round(f1, 4))

                    st.write("Matriu de confusió:")
                    st.write(confusion_matrix(y_test, y_pred))

                    # Guardar el model actualitzat
                    joblib.dump(nou_model, "model_geopay.pkl")
                    joblib.dump(list(X.columns), "columnes_model.pkl")

                    st.success("Model actualitzat i guardat correctament.")

                    st.warning(
                        "En Streamlit Cloud, aquest guardat pot no ser permanent si l'aplicació es reinicia. "
                        "En un entorn productiu real caldria guardar el model en un sistema persistent."
                    )

            except Exception as e:
                st.error(f"Error durant l'entrenament: {e}")