# pages/pipeline.py
"""
Page Streamlit : Pipeline d'analyse
- zone de saisie texte + bouton 'Analyser'
- choix du modèle fine-tuné (DistilBERT|BERT) dans la sidebar
- choix méthode explicabilité (Mistral|SHAP) dans la sidebar
- envoi du prompt à Mistral via call_mistral_api (fonction existante dans le projet)
- affichage des prédictions multilabel et du texte généré par Mistral
"""

import os
import json
import streamlit as st
import torch
import numpy as np
import traceback
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
)
import shap

#cleMistral= "uoTOdvSk0s14tC3POwuUx4X2Tw0nPKXl"

# import du style page (CSS)
from page_style import set_page_style

# CONFIG PAGE
st.set_page_config(layout="wide", page_title="Pipeline - AI Against Modern Slavery")
set_page_style()

# Détection device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mapping des modèles — on garde les clés originales
MODELS = {
    "DistilBERT": "distilbert-base-uncased",
    "BERT": "bert-base-uncased"
}
# Modèle Mistral (utilisé pour la génération explicative)
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistralai/Mistral-7B-Instruct")

# Labels multi-label (ordre important : correspond au fine-tuning)
LABELS = [
    "Approval","Signature","C1_reporting_entity","C2_structure","C2_operations",
    "C2_supply_chains","C3_risk_description","C4_risk_mitigation",
    "C4_remediation","C5_effectiveness","C6_consultation"
]

# -------------------------
# Sidebar : paramètres utilisateur
# -------------------------
with st.sidebar:
    st.header("⚙️ Paramètres")
    model_choice = st.selectbox("Choisir le modèle fine-tuné :", list(MODELS.keys()), index=0)
    pe_choice = st.selectbox("Méthode d’explicabilité :", ["Mistral", "SHAP"], "LIME", index=0)
    # Optionnel : réglages Mistral
    st.markdown("-----")
    st.write("Réglages Mistral (si utilisé)")
    mistral_max_tokens = st.number_input("Max tokens Mistral", min_value=50, max_value=2048, value=300)
    mistral_temperature = st.slider("Température Mistral", 0.0, 1.0, 0.0, 0.01)

# -------------------------
# Main UI : zone saisie + bouton
# -------------------------
st.title("⚙️ Pipeline d’analyse — Détection d’indices d’esclavage moderne")

st.markdown("Saisissez une phrase extraite d'une déclaration et cliquez sur **Analyser**. Le modèle multilabel prédit les critères pertinents ; Mistral (ou SHAP) fournit l'explication.")

# Zone de saisie texte (input="text")
sentence = st.text_area("Entrez une phrase à analyser :", "This statement was approved by the Board on 30 June 2021.")

# Bouton 'Valider' / 'Analyser'
analyze_btn = st.button("Analyser")

# Cache pour charger les tokenizers / modèles une seule fois par session
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_key):
    """
    Charge et renvoie tokenizer et modèle de classification (multi-label).
    Utilise cache_resource afin de ne pas recharger plusieurs fois.
    """
    model_name = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification"
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_mistral_model_and_tokenizer():
    """
    Charge le tokenizer et le modèle causal pour Mistral (si on veut exécuter localement).
    Sur Kaggle / environment restreint il est possible de ne pas charger Mistral localement,
    on préférera appeler l'API distante via call_mistral_api.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL)
        model = AutoModelForCausalLM.from_pretrained(MISTRAL_MODEL).to(DEVICE)
        model.eval()
        return tokenizer, model
    except Exception as e:
        # Si échec, on renvoie None pour indiquer qu'il faut utiliser l'API distante
        st.warning("Impossible de charger Mistral localement (fallback vers API distante si configurée).")
        return None, None

# --------------------------------------------------------------------------------------
# Fonction utilitaire : wrapper inference multi-label (prédictions et probabilités)
# --------------------------------------------------------------------------------------
def predict_multilabel(sentence, tokenizer, model, threshold=0.5):
    """
    Prend une phrase (string), renvoie (preds_int_array, probs_array)
    preds_int_array : Vecteur 0/1
    probs_array : probabilités pour chaque label
    """
    try:
        toks = tokenizer([sentence], return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            out = model(**toks)
            logits = out.logits  # shape (1, n_labels)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds = (probs >= threshold).astype(int)
        return preds, probs
    except Exception as e:
        st.error("Erreur lors de l'inférence modèle : " + str(e))
        st.exception(traceback.format_exc())
        return np.zeros(len(LABELS), dtype=int), np.zeros(len(LABELS))

# --------------------------------------------------------------------------------------
# Fonction d'appel à Mistral distante (utiliser la fonction globale call_mistral_api si fournie)
# --------------------------------------------------------------------------------------
# On suppose que ton projet contient déjà la fonction `call_mistral_api(prompt, ...)`
# Si non, tu peux importer / définir la version HTTP (voir docs Mistral). Ici on cherche
# à appeler une fonction du nom call_mistral_api si disponible dans le namespace global.
# --------------------------------------------------------------------------------------
def call_mistral_local_or_api(prompt, max_tokens=300, temperature=0.0, top_p=1.0):
    """
    Essaie d'utiliser une fonction call_mistral_api déjà définie (API distante).
    Sinon, si Mistral est chargé localement, exécute une génération locale.
    Retourne un texte généré.
    """
    # 1) si la fonction call_mistral_api existe globalement, l'utiliser (API distante)
    if "call_mistral_api" in globals():
        try:
            resp = globals()["call_mistral_api"](prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            # la fonction peut retourner dict { "generated_text": ..., "full_response": ... } ou un string
            if isinstance(resp, dict):
                return resp.get("generated_text") or json.dumps(resp)
            elif isinstance(resp, str):
                return resp
            else:
                return str(resp)
        except Exception as e:
            st.warning("Appel à call_mistral_api a échoué : " + str(e))

    # 2) sinon tenter génération locale (si le modèle Mistral local est chargé)
    ml_tok, ml_model = load_mistral_model_and_tokenizer()
    if ml_tok is not None and ml_model is not None:
        try:
            inputs = ml_tok(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = ml_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=(temperature > 0.0), temperature=float(temperature))
            gen = ml_tok.decode(out[0], skip_special_tokens=True)
            return gen
        except Exception as e:
            st.warning("Échec génération locale Mistral : " + str(e))

    # 3) fallback texte simulé
    return "MODE_SIMULATION: Aucune API Mistral disponible. Remplacez par un appel réel à call_mistral_api(prompt)."

# --------------------------------------------------------------------------------------
# Bouton cliqué : exécution du pipeline
# --------------------------------------------------------------------------------------
if analyze_btn:
    st.info("Lancement de l'analyse... (cela peut prendre quelques secondes)")

    # Charger modèle/tokenizer (cached)
    try:
        tokenizer, model = load_model_and_tokenizer(model_choice)
    except Exception as e:
        st.error("Impossible de charger le modèle choisi : " + str(e))
        st.stop()

    # Prédiction multilabel
    preds, probs = predict_multilabel(sentence, tokenizer, model, threshold=0.5)

    # Affichage clair des résultats
    st.subheader("🔎 Prédictions multilabel")
    # Construire un dictionnaire affichable : label -> prob
    results = { LABELS[i]: float(probs[i]) for i in range(len(LABELS)) if preds[i] == 1 }
    if len(results) == 0:
        st.warning("Aucune étiquette prédite comme pertinente (seuil 0.5).")
    else:
        st.success("Critères détectés :")
        st.json(results)

    # Explicabilité :
    st.subheader("🧾 Explication / justification")

    if pe_choice == "Mistral":
        # Construire un prompt simple enrichi (tu peux brancher ici ton RAG builder)
        # On envoie la phrase + labels candidates + probabilités
        pred_labels = [LABELS[i] for i in range(len(LABELS)) if preds[i] == 1]
        prompt = (
            "Tu es un expert en conformité et modern slavery. "
            f"Phrase : {sentence}\n\n"
            f"Prédictions modèles : {json.dumps({LABELS[i]: float(probs[i]) for i in range(len(LABELS))})}\n\n"
            "Indique clairement si la phrase constitue un cas pertinent lié à l'esclavage moderne, "
            "précise sur quel critère(s) et justifie en 2-4 phrases. "
            "Fournis une suggestion d'action pour un analyste (1 phrase). Réponds en français."
        )
        # Appel à Mistral (API ou local)
        gen_text = call_mistral_local_or_api(prompt, max_tokens=mistral_max_tokens, temperature=mistral_temperature)
        # Afficher le texte généré dans une zone 'label' (ici text_area en lecture seule)
        st.text_area("📝 Explication Mistral :", gen_text, height=240)

    elif pe_choice == "SHAP":
        # SHAP explainer (simplifié)
        try:
            def model_predict_for_shap(texts):
                toks = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    logits = model(**toks).logits
                    return torch.sigmoid(logits).cpu().numpy()
            explainer = shap.Explainer(model_predict_for_shap, tokenizer, output_names=LABELS)
            shap_values = explainer([sentence])
            st.write("Résultat SHAP (valeurs):")
            st.write(shap_values.values)  # peut être un array multi-dim
            st.write("Pour visualiser SHAP, utilisez shap.plots.* dans un environnement notebook.")
        except Exception as e:
            st.error("Erreur lors du calcul SHAP : " + str(e))
            st.exception(traceback.format_exc())

# -------------------------
# Footer
# -------------------------
st.markdown("""
    <div class="footer">
        © 2025 New Horizons Foundation 🌍 | AI Against Modern Slavery
    </div>
""", unsafe_allow_html=True)