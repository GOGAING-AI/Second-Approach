import streamlit as st
import os
from page_style import set_page_style  # import depuis page_style.py

# ========= CONFIG APP =========
st.set_page_config(
    page_title="AI against Modern Slavery - AIMS Hackathon",
    layout="wide"
)

# ========= CSS GLOBAL =========


# Appel du style global (header + background)
set_page_style()

# ========= CONTENU =========
st.markdown("""
### 👋 Bienvenue sur notre application !

Utilisez la **barre latérale** pour naviguer entre :
- 🏠 **Accueil** : Présentation de l’équipe et de la solution  
- 📊 **Visualisation** : Diagrammes exploratoires des données  
- ⚙️ **Pipeline** : Testez DistilBERT/BERT avec Mistral + LIME/SHAP + RAG  
""")

# --------------------------- Pied de page --------------------------------
st.markdown("""
    <div class="footer">
        © 2025 New Horizons Foundation 🌍 | AI Against Modern Slavery
    </div>
""", unsafe_allow_html=True)
