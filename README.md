# 🌍 AI against Modern Slavery - AIMS Hackathon

## 🎯 Objectif du projet
Ce projet a été développé dans le cadre du **AIMS Hackathon** :  
**AI against Modern Slavery**.  

Notre solution a pour but d’analyser automatiquement des rapports d’entreprises afin de détecter les indices de **travail forcé et d’esclavage moderne**, en s’appuyant sur des **modèles de NLP** (LLMs) et des méthodes d’explicabilité.  

---

## 🏗️ Architecture de l’application
L’application a été construite avec **Streamlit** et suit une organisation **multi-pages** :

app/
├── app.py # Point d’entrée principal
├── pages/
│ ├── accueil.py # Présentation équipe + logos + briefing
│ ├── visualisation.py # Visualisation des données (EDA)
│ └── pipeline.py # Test du pipeline (DistilBERT, BERT, LLaMA + Mistral/LIME/SHAP RAG)
├── assets/
│ ├── team/ # Photos des membres de l’équipe
│ ├── logos/ # Logos équipe et hackathon
│ |── plots/ # Diagrammes sauvegardés
| └──images/ #image de Background
├── outputs/ # Modèles fine-tunés et résultats
└── train_aims.csv # Dataset de travail

## ⚙️ Fonctionnalités principales
1. **Accueil**
   - Présentation du projet et de l’équipe *New Horizons Foundation*.
   - Logos (équipe + hackathon).
   - Objectif de la solution.

2. **Visualisation**
   - Analyse exploratoire du dataset `train_aims.csv`.
   - Diagrammes générés :
     - Distribution des critères d’annotation.
     - Distribution de la longueur des phrases.
     - Matrice de corrélation entre critères.

3. **Pipeline**
   - Choix du modèle d’entraînement :
     - **DistilBERT**
     - **BERT**
     - **LLaMA**
   - Choix de la méthode d’explicabilité :
     - **Mistral (RAG)**
     - **LIME**
     - **SHAP**
   - Entrée d’une phrase → sortie :
     - Prédictions des critères (11 labels multilabels).
     - Explication via l’approche choisie.
     - Contexte supplémentaire (via RAG pour Mistral).

---

## 🧰 Technologies utilisées
- **Python 3.10+**
- **Streamlit** (interface)
- **Transformers (Hugging Face)** pour DistilBERT, BERT, LLaMA et Mistral
- **Sentence-Transformers** pour les embeddings
- **FAISS** pour l’indexation RAG
- **scikit-learn** pour les métriques
- **SHAP** et **LIME** pour l’explicabilité
- **Seaborn / Matplotlib** pour la visualisation

---

## 🚀 Installation et lancement

- a. Installer les dépendances
     pip install -r requirements.txt

- b. Lancer l’application
     streamlit run app.py

- c. L’interface sera disponible sur :
    👉 http://localhost:8501