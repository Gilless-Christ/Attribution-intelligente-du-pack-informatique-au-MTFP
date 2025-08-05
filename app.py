# ---- Fonctions utiles ----
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def remplacer_illisible_par_nan(df):
    return df.replace(r'(?i)^\s*illisible\s*$', np.nan, regex=True)

def calculer_age_equipement(df, colonne_annee="Année d'acquisition", colonne_resultat="Âge_équipement", annee_reference=2025):
    df[colonne_annee] = pd.to_numeric(df[colonne_annee], errors='coerce')
    df[colonne_resultat] = annee_reference - df[colonne_annee]
    return df

def afficher_valeurs_uniques_colonne(df, colonne):
    valeurs_uniques = df[colonne].value_counts(dropna=False)

    # Affichage texte
    st.write(f"### 📊 Valeurs uniques dans la colonne : `{colonne}`")
    st.write(valeurs_uniques)

    # Visualisation graphique
    fig, ax = plt.subplots(figsize=(7, 4))
    valeurs_uniques.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    ax.set_title(f'Valeurs uniques - {colonne}', fontsize=14)
    ax.set_xlabel('Valeur', fontsize=12)
    ax.set_ylabel('Fréquence', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)



# ---- Interface utilisateur ----
st.title("📊 Analyse intelligente du pack informatique (MTFP)")

uploaded_file = st.file_uploader("📁 Uploadez un fichier CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 Aperçu des données")
    st.dataframe(df.head())

    

    # Nettoyage et calcul de l'âge
    df = remplacer_illisible_par_nan(df)
    st.subheader("✅ Sélection de la colonne Année d'acquisition")
    colonne_annee = st.selectbox("🗓️ Sélectionnez la colonne de l'année d'acquisition", df.columns)
    df = calculer_age_equipement(df, colonne_annee)
 

  

    st.subheader("📈 Données après traitement")
    st.dataframe(df)

    # Visualisation des valeurs uniques d'une colonne
    st.subheader("🔍 Visualisation des valeurs uniques par colonne")
    colonne_a_visualiser = st.selectbox("📌 Sélectionnez une colonne à analyser", df.columns)
    if colonne_a_visualiser:
        afficher_valeurs_uniques_colonne(df, colonne_a_visualiser)

    # Statistiques de base
    st.subheader("📊 Statistiques de l'âge des équipements")
    st.write(df["Âge_équipement"].describe())

    