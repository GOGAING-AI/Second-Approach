# page_style.py
from pathlib import Path
import streamlit as st
import os


BASE_DIR = Path(__file__).resolve().parent

def asset_path(*parts):
    """Renvoie un chemin relatif vers assets, sous forme de string.
       Ex: asset_path('logos','hackathon_logo.png')"""
    
    return str(BASE_DIR.joinpath(*parts))

def set_page_style(bg_image="assets/images/modernSalve0.png", logo="assets/logos/hackathon_logo.png"):
    """
    Applique le CSS global et affiche un header stylé.
    - bg_image, logo : chemins relatifs sous le répertoire du projet.
    """
    # Vérifie la présence des images
    bg = asset_path(bg_image)
    logo_path = asset_path(logo)
    bg_exists = Path(bg).exists()
    logo_exists = Path(logo_path).exists()

    """
    Applique un style global via st.markdown(unsafe_allow_html=True).
    Modifier ici pour changer la charte visuelle.
    """
    primary = "#1a3d6d"   # couleur principale
    accent = "#ff6b6b"    # accent si besoin

    # CSS : background cover (plein écran pour l'image), header à gauche
    css = f"""
    <style>
    .stApp {{
        {"background-image: url('" + bg + "');" if bg_exists else "background-color: #f5f7fa;"}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}

            
     #/* Contenu lisible sur le fond */
    .main {{
    
        background: rgba(255, 255, 255, 0.88);
        padding: 20px;
        border-radius: 12px;
        margin: 20px;

     }}


    .header-container {{
        display:flex;
        align-items:center;
        gap:16px;
        background: rgba(44,62,80,0.88);
        padding: 14px;
        border-radius: 10px;
        color: #fff;
        margin-bottom: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }}
    .header-logo {{
        width: 110px;
        height: auto;
        border-radius: 8px;
        background: #fff;
        padding: 6px;
    }}
    .header-title h1{{ margin:0; font-size:22px; line-height:1; }}
    .header-title p{{ margin:0; color:#ecf0f1; }}
    .team-card {{ text-align:center; padding:12px; border-radius:10px; background:rgba(255,255,255,0.96); box-shadow:0 6px 14px rgba(0,0,0,0.06); }}
    .footer {{ text-align:center; margin-top:30px; font-size:13px; color:#666; padding:12px 0; border-top:1px solid #eee; }}
     <style>
    /* Page background */
    .stApp {{
        background-color: #f7f9fc;
        color: #0f1724;
    }}
    /* Titres */
    .css-1v3fvcr h1, .stTitle h1 {{
        color: {primary};
    }}
    /* Sidebar style */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #eef2f7, #ffffff);
        border-radius: 8px;
        padding: 12px;
    }}
    /* Footer box */
    .footer {{
        margin-top: 40px;
        padding: 12px;
        text-align: center;
        color: white;
        background: {primary};
        border-radius: 8px;
        font-size: 14px;
    }}
    /* Small helpers */
    .stButton>button {{
        background-color: {primary};
        color: white;
    }}

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Header HTML (logo à gauche + titre)

    logo_html = f'<img src="{logo}" class="header-logo"/>' if logo_exists else ""
    
    header_html = f"""
    <div class="header-container">
      {logo_html}
      <div class="header-title">
        <h1>AI against Modern Slavery</h1>
        <p>AIMS Hackathon — New Horizons Foundation</p>
      </div>
    </div>

    """
    st.markdown(header_html, unsafe_allow_html=True)
