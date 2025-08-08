import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import base64

# Configuration de la page
st.set_page_config(page_title="Renouvellement Pack Informatique - MTFP", layout="wide")

# Fonction pour charger les données
@st.cache_data
def charger_donnees(fichier):
    df = pd.read_csv(fichier)
    df.replace("illisible", np.nan, inplace=True)
    return df

# Fonction pour calculer l'âge
def calculer_age(df, colonne_date, nouvelle_colonne="Âge de l'équipement"):
    df = df.copy()
    df[nouvelle_colonne] = pd.to_datetime(df[colonne_date], errors="coerce")
    df[nouvelle_colonne] = datetime.now().year - df[nouvelle_colonne].dt.year
    return df

# Fonction pour prédire le renouvellement
def predire_renouvellement(df):
    df = df.copy()
    def besoin_renouvellement(row):
        if pd.isna(row.get("Âge de l'équipement")) or pd.isna(row.get("Catégorie d'utilisateur")):
            return "Inconnu"
        if row["Âge de l'équipement"] >= 4:
            return "Oui"
        else:
            return "Non"
    df["À renouveler"] = df.apply(besoin_renouvellement, axis=1)
    return df

# Fonction d'export CSV
def telecharger_csv(df):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="donnees_exportees.csv">📥 Télécharger les résultats</a>'
    return href

# Fonction d'entraînement modèle ML
def entrainer_modele(df):
    df_modele = df[["Âge de l'équipement", "Catégorie d'utilisateur", "À renouveler"]].dropna()
    df_modele = df_modele.copy()
    df_modele["Catégorie d'utilisateur"] = df_modele["Catégorie d'utilisateur"].astype("category").cat.codes
    df_modele["À renouveler"] = df_modele["À renouveler"].map({"Oui": 1, "Non": 0})

    X = df_modele[["Âge de l'équipement", "Catégorie d'utilisateur"]]
    y = df_modele["À renouveler"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modele = RandomForestClassifier(n_estimators=100, random_state=42)
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)
    rapport = classification_report(y_test, y_pred, output_dict=True)
    return modele, rapport

# Interface utilisateur principale
st.title("💻 Système intelligent de gestion du renouvellement des packs informatiques - MTFP")

# Chargement des données
fichier = st.sidebar.file_uploader("📁 Importer un fichier CSV", type="csv")

if fichier is not None:

    if "df" not in st.session_state:
        st.session_state.df = charger_donnees(fichier)

    df = st.session_state.df

    # Choix de l'action dans la barre latérale
    menu = st.sidebar.radio("🧭 Menu", [
        "Aperçu des données",
        "Nettoyage et Prétraitement",
        "Statistiques descriptives",
        "Détection des appareils à renouveler",
        "Modèle de prédiction (ML)",
        "Export des données"
    ])

    if menu == "Aperçu des données":
        st.subheader("🔍 Données brutes importées")
        st.dataframe(df)

    elif menu == "Nettoyage et Prétraitement":
        colonne_date = st.selectbox("📅 Sélectionner la colonne de date d'achat", df.columns)
        df = calculer_age(df, colonne_date)
        st.session_state.df = df  # Mise à jour du dataframe dans session_state
        st.success("✅ Âge de l’équipement calculé")
        st.dataframe(df.head())

    elif menu == "Statistiques descriptives":
        st.subheader("📊 Statistiques descriptives")
        st.write(df.describe(include='all'))
        st.write("Distribution de l'âge des équipements")
        if "Âge de l'équipement" in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df["Âge de l'équipement"].dropna(), bins=10, kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Veuillez d’abord calculer l'âge de l'équipement dans la section Nettoyage et Prétraitement.")

    elif menu == "Détection des appareils à renouveler":
        if "Âge de l'équipement" not in df.columns:
            st.warning("Veuillez d’abord calculer l'âge de l'équipement dans la section Nettoyage et Prétraitement.")
        else:
            df = predire_renouvellement(df)
            st.session_state.df = df
            st.subheader("📌 Appareils à renouveler")
            st.dataframe(df[df["À renouveler"] == "Oui"])
            st.metric("Nombre d'appareils à renouveler", df["À renouveler"].value_counts().get("Oui", 0))

    elif menu == "Modèle de prédiction (ML)":
        if "Âge de l'équipement" not in df.columns:
            st.warning("Veuillez d’abord calculer l'âge de l'équipement dans la section Nettoyage et Prétraitement.")
        else:
            st.subheader("🧠 Prédiction avec modèle de Machine Learning")
            df = predire_renouvellement(df)
            st.session_state.df = df
            modele, rapport = entrainer_modele(df)
            st.success("✅ Modèle entraîné")
            st.json(rapport)

    elif menu == "Export des données":
        if "À renouveler" in df.columns:
            st.markdown(telecharger_csv(df), unsafe_allow_html=True)
        else:
            st.warning("Veuillez d’abord exécuter la détection de renouvellement.")
else:
    st.info("📁 Importez un fichier CSV pour commencer.")
