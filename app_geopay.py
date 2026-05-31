import streamlit as st
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# --------------------------------------------------
# Configuració visual de la pàgina
# --------------------------------------------------
st.set_page_config(
    page_title="GeoPay",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }

    .title-box {
        background-color: #0f172a;
        padding: 26px;
        border-radius: 14px;
        margin-bottom: 24px;
    }

    .title-box h1 {
        color: white;
        margin: 0;
        font-size: 36px;
    }

    .title-box p {
        color: #cbd5e1;
        margin-top: 8px;
        font-size: 17px;
    }

    .section-card {
        background-color: white;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 18px;
    }

    .risk-ok {
        background-color: #ecfdf5;
        border-left: 7px solid #10b981;
        padding: 18px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: 600;
        color: #065f46;
        margin-bottom: 14px;
    }

    .risk-alert {
        background-color: #fef2f2;
        border-left: 7px solid #ef4444;
        padding: 18px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: 600;
        color: #991b1b;
        margin-bottom: 14px;
    }

    .small-note {
        color: #64748b;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Càrrega del model existent
# --------------------------------------------------
model = joblib.load("model_geopay.pkl")
columnes_model = joblib.load("columnes_model.pkl")


# --------------------------------------------------
# Funció de preprocessament per entrenament
# --------------------------------------------------
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


# --------------------------------------------------
# Capçalera
# --------------------------------------------------
st.markdown("""
<div class="title-box">
    <h1>GeoPay</h1>
    <p>Monitor de risc transaccional per detectar possibles operacions fraudulentes</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Pestanyes principals
# --------------------------------------------------
tab_prediccio, tab_entrenament = st.tabs([
    "Predicció d'operació",
    "Entrenament del model"
])


# --------------------------------------------------
# Pestanya 1: Predicció
# --------------------------------------------------
with tab_prediccio:

    st.markdown("### Consulta d'una operació bancària")
    st.markdown(
        '<p class="small-note">Introdueix les dades de la transacció per obtenir una estimació del risc de frau.</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Dades del compte origen")
        amount = st.number_input("Import de l'operació", min_value=0.0, value=30000.0)
        oldbalanceOrig = st.number_input("Saldo origen abans", min_value=0.0, value=30000.0)
        newbalanceOrig = st.number_input("Saldo origen després", min_value=0.0, value=0.0)

    with col2:
        st.markdown("#### Dades del compte destí")
        oldbalanceDest = st.number_input("Saldo destí abans", min_value=0.0, value=0.0)
        newbalanceDest = st.number_input("Saldo destí després", min_value=0.0, value=30000.0)
        tipus_operacio = st.selectbox(
            "Tipus d'operació",
            ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
        )

    st.divider()

    if st.button("Analitzar operació", type="primary"):

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

        col_resultat, col_detall = st.columns([1, 1])

        with col_resultat:
            st.markdown("### Resultat de GeoPay")

            if prediccio == 1:
                st.markdown(
                    '<div class="risk-alert">Operació sospitosa: possible frau</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="risk-ok">Operació aparentment normal</div>',
                    unsafe_allow_html=True
                )

            st.metric("Probabilitat estimada de frau", f"{probabilitat * 100:.2f}%")

        with col_detall:
            st.markdown("### Resum de l'operació")
            st.metric("Diferència saldo origen", f"{balanceOrig_diff:,.2f}")
            st.metric("Diferència saldo destí", f"{balanceDest_diff:,.2f}")
            st.metric("Import elevat", "Sí" if high_amount == 1 else "No")

        with st.expander("Veure dades utilitzades pel model"):
            st.dataframe(entrada)

        st.warning(
            "Aquesta predicció és orientativa i hauria de ser revisada per un equip humà en casos sensibles."
        )


# --------------------------------------------------
# Pestanya 2: Entrenament
# --------------------------------------------------
with tab_entrenament:

    st.markdown("### Actualització del model amb noves dades")
    st.markdown(
        '<p class="small-note">Puja un fitxer CSV amb dades de transaccions per entrenar una nova versió del model GeoPay.</p>',
        unsafe_allow_html=True
    )

    fitxer_csv = st.file_uploader(
        "Selecciona un fitxer CSV",
        type=["csv"]
    )

    if fitxer_csv is not None:

        df_nou = pd.read_csv(fitxer_csv)

        col_info1, col_info2 = st.columns(2)

        with col_info1:
            st.metric("Files carregades", df_nou.shape[0])

        with col_info2:
            st.metric("Columnes carregades", df_nou.shape[1])

        with st.expander("Vista inicial del dataset"):
            st.dataframe(df_nou.head())

        st.divider()

        if st.button("Entrenar nou model", type="primary"):

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

                    st.markdown("### Mètriques del nou model")

                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

                    with col_m1:
                        st.metric("Accuracy", f"{accuracy:.4f}")

                    with col_m2:
                        st.metric("Precision", f"{precision:.4f}")

                    with col_m3:
                        st.metric("Recall", f"{recall:.4f}")

                    with col_m4:
                        st.metric("F1-score", f"{f1:.4f}")

                    st.markdown("### Matriu de confusió")
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