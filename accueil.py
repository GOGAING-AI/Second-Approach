import streamlit as st
from pathlib import Path
from PIL import Image
import os
import app  # Pour le style global
from page_style import set_page_style  # import depuis page_style.py

# ========= CONFIG =========
st.set_page_config(layout="wide")

#st.title("🏆 AIMS Hackathon - Accueil")
#st.subheader("AI against Modern Slavery")

set_page_style()  # applique la même entête & background

# Base dir du projet (deux niveaux au-dessus de pages)
BASE_DIR = Path(__file__).resolve().parent.parent

st.title("🏆 AIMS Hackathon - Accueil")
st.subheader("AI against Modern Slavery")

# Afficher logos (colonnés)
col1, col2 = st.columns([1,1])
team_logo = BASE_DIR / "assets" / "logos" / "team_logo.jpeg"
hackathon_logo = BASE_DIR / "assets" / "logos" / "hackathon_logo.png"

with col1:
    if team_logo.exists():
        st.image(str(team_logo), caption="Logo de l'équipe", use_column_width=True)
    else:
        st.warning("Logo d'équipe introuvable : " + str(team_logo))
with col2:
    if hackathon_logo.exists():
        st.image(str(hackathon_logo), caption="Logo Hackathon", use_column_width=True)
    else:
        st.warning("Logo hackathon introuvable : " + str(hackathon_logo))

# ========= Image de fond spécifique Accueil =========
st.markdown("""
    <style>
        .stApp {
            background-image: url("assets/images/modernSalve0.png");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }
    </style>
""", unsafe_allow_html=True)


# ========= Équipe =========
st.header("👨‍💻 Équipe New Horizons Foundation")

team = [
    {"name": "Gogaing Foguen Baudouin Legrand", "role": "Chef d’équipe", "level": "MSc Research - Mecatronics Engineer & Data Scientist", "photo": os.path.join(BASE_DIR, "assets", "team", "Gogaing_Foguen_Baudouin_Legrand.jpeg")},
    {"name": "Tchakam Duplex Cedric", "role": "Data Scientist", "level": "Master 1 - Embedded systems and AI", "photo": os.path.join(BASE_DIR, "assets", "team", "Tchakam_Duplex_Cedric.jpeg")},
    {"name": "Noupoue Mbouga Desvan-Kerol", "role": "ML Engineer", "level": "Master 1 - Embedded systems and AI", "photo": os.path.join(BASE_DIR, "assets", "team", "Noupoue_Mbouga_Desvan_Kerol.jpeg")},
    {"name": "KEUTCHA TOALEU JOËL", "role": "Data Scientist", "level": "Master 1 - Embedded systems and AI", "photo": os.path.join(BASE_DIR, "assets", "team", "KEUTCHA_TOALEU_JOËL.jpeg")},
    {"name": "Foka Maghem Yann Brondon", "role": "ML Engineer", "level": "Master 1 - Embedded systems and AI", "photo": os.path.join(BASE_DIR, "assets", "team", "Foka_Maghem_Yann_Brondon.jpeg")}
]

        

cols = st.columns(len(team))
for i, member in enumerate(team):
    with cols[i]:
        # Utilisez st.image() pour afficher la photo
        # Le chemin d'accès est déjà correct grâce à os.path.join
        st.image(member['photo'], width=120)
        st.markdown(f"""
        <div class="team-card">
            <img src="{member['photo']}" width="120" style="border-radius:50%; margin-bottom:10px;">
            <h4>{member['name']}</h4>
            <p><em>{member['role']}</em></p>
            <p style="font-size:13px; color:gray;">{member['level']}</p>
        </div>
        """, unsafe_allow_html=True)

      
       # Utilisez st.markdown() ou st.write() pour le texte
        #st.markdown(f"<h4>{member['name']}</h4>", unsafe_allow_html=True)
        #st.markdown(f"<p><em>{member['role']}</em></p>", unsafe_allow_html=True)
        #st.markdown(f"<p style='font-size:13px; color:gray;'>{member['level']}</p>", unsafe_allow_html=True)

       
# --- Footer ---

st.markdown("""
    <div class="footer">
        © 2025 New Horizons Foundation 🌍 | AI Against Modern Slavery
    </div>
""", unsafe_allow_html=True)



