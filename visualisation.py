# pages/visualisation.py
"""
Page Streamlit : Visualisation
- EDA déjà existant conservé (ta logique)
- Ajout d'une galerie 3x4 (12 images) présentées en grille avec CSS
- Possibilité d'uploader des images supplémentaires (si besoin)
"""

import os
import csv
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from page_style import set_page_style

#cleMistral= "uoTOdvSk0s14tC3POwuUx4X2Tw0nPKXl"

# Configuration page
st.set_page_config(layout="wide", page_title="Visualisations")
set_page_style()

# Background image (optionnel)
st.markdown("""
    <style>
        .visual-bg {
            background-image: url('assets/images/modernSalve1.png');
            background-size: cover;
            background-position: center;
            opacity: 0.06;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Visualisation des données")

# --- CHEMIN DU CSV (A ADAPTER) ---
# On définit le chemin par défaut pour qu'il pointe vers le fichier dans le même dossier
AU_FILE_DEFAULT = "train_aims.csv"
au_file = st.text_input("Chemin du dataset (CSV)", AU_FILE_DEFAULT)
# Chargement simple et robuste du CSV
@st.cache_data(show_spinner=False)
def load_df(path):
    try:
        # Tentative lecture avec pandas (détection automatique du séparateur)
        df = pd.read_csv(path, engine="python")
        # Nettoyage minimal
        df.dropna(subset=["sentence"], inplace=True)
        # Forcer existence colonnes de critères si manquantes -> crée colonnes à 0
        expected_cols = [
            'Approval', 'Signature', 'c1_reporting_entity', 'c2_structure', 'c2_operation',
            'c2_supply_chains', 'c3_risk_description', 'c4_mitigation', 'c4_remediation',
            'c5_effectiveness', 'c6_consultation'
        ]
        for c in expected_cols:
            if c not in df.columns:
                df[c] = 0
        # Calculs utiles
        df["word_count"] = df["sentence"].astype(str).apply(lambda s: len(s.split()))
        df["char_count"] = df["sentence"].astype(str).apply(lambda s: len(s))
        df["label_count"] = df[[c for c in expected_cols]].sum(axis=1)
        return df
    except Exception as e:
        st.error("Erreur lecture CSV: " + str(e))
        return None

df = load_df(au_file)
if df is None:
    st.warning("Aucun dataset valide chargé. Vérifiez le chemin et le format du CSV.")
    st.stop()

# ====================================================
#  Visualisations : on préserve la logique d'origine
# ====================================================
st.header("Exploration (extraits)")

# 1. Distribution longueur en mots
fig1, ax1 = plt.subplots(figsize=(8,4))
sns.histplot(df["word_count"], bins=50, kde=True, color="purple", ax=ax1)
ax1.set_title("Distribution du nombre de mots par phrase")
st.pyplot(fig1)
plt.close(fig1)
plt.show()

# 2. Distribution longueur en caractères
fig2, ax2 = plt.subplots(figsize=(8,4))
sns.histplot(df["char_count"], bins=50, kde=True, color="teal", ax=ax2)
ax2.set_title("Distribution du nombre de caractères par phrase")
st.pyplot(fig2)
plt.close(fig2)
plt.show()

# 3. Histogramme des labels (somme par colonne)
criteria_cols = [
    'Approval', 'Signature', 'c1_reporting_entity', 'c2_structure', 'c2_operation',
    'c2_supply_chains', 'c3_risk_description', 'c4_mitigation', 'c4_remediation',
    'c5_effectiveness', 'c6_consultation'
]
fig3, ax3 = plt.subplots(figsize=(10,6))
df[criteria_cols].sum().sort_values(ascending=True).plot(kind="barh", color="orange", ax=ax3)
ax3.set_title("Distribution des labels (multi-label)")
st.pyplot(fig3)
plt.close(fig3)
plt.show()

# 4. WordCloud
st.subheader("Nuage de mots (extrait)")
text_all = " ".join(df["sentence"].astype(str).tolist())
wc = WordCloud(width=900, height=400, background_color="white").generate(text_all[:200000])
fig4, ax4 = plt.subplots(figsize=(10,4))
ax4.imshow(wc, interpolation="bilinear")
ax4.axis("off")
st.pyplot(fig4)
plt.close(fig4)

# ---------------------------------------------------------
# Galerie : 12 images en 3 lignes x 4 colonnes (editable/upload)
# ---------------------------------------------------------
st.header("Galerie des diagrammes")
st.markdown("Vous pouvez uploader jusqu'à 12 images (3×4) ou utiliser les images présentes dans `assets/plots/`.")

# Upload multiple images (optionnel)
uploaded = st.file_uploader("Uploader des images (optionnel, multi)", accept_multiple_files=True, type=["png","jpg","jpeg"])
images_to_show = []

# Si l'utilisateur upload, on priorise ces images
if uploaded:
    for f in uploaded[:12]:
        # on lit comme bytes et on affiche ensuite
        images_to_show.append(f)
else:
    # sinon on récupère 12 images depuis assets/plots/ (si elles existent)
    static_plots_dir = "assets/plots/"
    if os.path.exists(static_plots_dir):
        files = sorted([os.path.join(static_plots_dir, p) for p in os.listdir(static_plots_dir) if p.lower().endswith((".png",".jpg",".jpeg"))])
        images_to_show = files[:12]

# CSS pour la galerie (3x4)
st.markdown("""
    <style>
    .grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 18px;
      padding: 8px;
    }
    .grid img {
      width: 100%;
      height: auto;
      border-radius: 10px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.12);
      transition: transform .25s ease;
    }
    .grid img:hover { transform: scale(1.03); }
    </style>
""", unsafe_allow_html=True)

# Build HTML
html = "<div class='grid'>"
if images_to_show:
    for im in images_to_show:
        # If it's an uploaded file, we can show via st.image directly; otherwise give an img tag path
        if hasattr(im, "read"):
            # uploaded file-like
            b = im.read()
            import base64
            enc = base64.b64encode(b).decode()
            html += f"<div><img src='data:image/png;base64,{enc}' /></div>"
        else:
            # path string
            html += f"<div><img src='/{im}' /></div>"
else:
    html += "<div>Aucune image trouvée (upload une image ou placez des plots dans assets/plots/)</div>"
html += "</div>"

st.markdown(html, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        © 2025 New Horizons Foundation 🌍 | AI Against Modern Slavery
    </div>
""", unsafe_allow_html=True)