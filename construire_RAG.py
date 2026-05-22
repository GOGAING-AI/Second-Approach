

# -----------------------------------------------------------------------------------
# 11) RAG simple : indexer AIMSPrompt.DOCX et aims_flattened_criteria.csv via TF-IDF
# - objectif : récupérer passages pertinents (top-k) pour une phrase classifiée
# -----------------------------------------------------------------------------------

# ==== CONFIGURATION / CHEMINS ====
DATA_DIR = Path("/kaggle/input")
INPUT_CSV = DATA_DIR / "aims-hackathon/train_aims.csv"           # dataset original
AIMS_DOCX = DATA_DIR / "document/AIMSPrompt.DOCX"          # docx contenant les templates
DATA_DIR_OUT = "/kaggle/working/"
OUTPUT_PHRASE_CSV = DATA_DIR_OUT / "phrase_pertinante.csv"
RAG_INDEX_PATH = DATA_DIR_OUT / "rag/rag_index.npz"       # optionnel pour sauvegarder le TF-IDF
RAG_META_PATH = DATA_DIR_OUT / "rag/rag_meta.json"

# Critères d'intérêt (doivent correspondre aux tokens dans AIMSPrompt.DOCX)
CRITERIA = [
    "Approval", "Signature", "Reporting entity", "Structure", "Operations",
    "Supply chains", "Risk description", "Risk mitigation", "Remediation",
    "Effectiveness", "Consultation"
]

# Normalisation basique d'une clé critère (pour comparer)
def normalize_key(s):
    return re.sub(r'\s+',' ', str(s).strip().lower())

CRITERIA_NORM = {normalize_key(c): c for c in CRITERIA}

# ==== 1) Création du fichier phrase_pertinante.csv ====
def create_phrase_pertinante_csv(input_csv_path, output_csv_path, criteria_columns):
    """
    Parcourt train_aims.csv et crée phrase_pertinante.csv avec colonnes:
      - criterion : nom du critère (ex: 'Approval', 'c3_risk_description' etc.)
      - sentence  : texte de la phrase
      - valeur    : 0 ou 1 (on inclut uniquement 0 ou 1)
    """
    print("Chargement du dataset source :", input_csv_path)
    # Détection automatique du séparateur ; plus simple : pandas saura souvent lire
    df = pd.read_csv(input_csv_path, sep=None, engine="python")
    print("Shape source:", df.shape)

    rows = []
    # On normalise mapping colonne->critère lisible si nécessaire
    for col in criteria_columns:
        if col not in df.columns:
            raise ValueError(f"Colonne attendue non trouvée dans le CSV: {col}")

    for crit in criteria_columns:
        # prendre uniquement lignes où crit est 0 ou 1
        sub = df.loc[df[crit].isin([0,1]), ["sentence", crit]].copy()
        sub = sub.rename(columns={crit: "valeur"})
        sub["criterion"] = crit
        rows.append(sub[["criterion","sentence","valeur"]])
    flat = pd.concat(rows, ignore_index=True)
    # Réordonner colonnes
    flat = flat[["criterion","sentence","valeur"]]
    flat.to_csv(output_csv_path, index=False, encoding="utf-8")
    print("Fichier phrase_pertinante.csv créé:", output_csv_path, "lignes:", len(flat))
    return flat

#----------------------------------------------------------------------------------

#-----------------------------------------------------
# ==== 12) Parser AIMSPrompt.DOCX pour extraire templates no-context / with-context ====

def parse_aims_docx(docx_path):
    """
    Parse AIMSPrompt.DOCX et retourne dict:
      contexts = {
        'Approval': {'no-context': [str paragraphe...], 'with-context': [str paragraphe...]},
        'Signature': ...
      }
    Méthode :
    - lit paragraphes du docx
    - repère les sections "Prompt template (no-context)" et "Prompt template (with-context)"
    - quand un critère (mot exact ou forme) est trouvé dans un paragraphe unique, on associe
      le bloc de paragraphe suivant (ou le paragraphe courant) comme contexte.
    """
    contexts = defaultdict(lambda: {"no-context": [], "with-context": []})

    if not DOCX_AVAILABLE or not docx_path.exists():
        print("AIMSPrompt.DOCX introuvable ou python-docx non installé. Retourne contexte vide.")
        return contexts

    doc = DocxDocument(str(docx_path))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip() != ""]
    n = len(paras)
    # repérer indices des headers 'Prompt template (no-context)' et 'Prompt template (with-context)'
    # On parcourt et lorsqu'on trouve un header, on lit les paragraphes suivants jusqu'au prochain header vide ou critère
    header_pattern_no = re.compile(r"prompt template\s*\(?no[-\s]*context\)?", flags=re.I)
    header_pattern_with = re.compile(r"prompt template\s*\(?with[-\s]*context\)?", flags=re.I)

    # Pour efficacité, construire mapping nom_critère_normalisé -> canonical
    crit_keys = {normalize_key(k): k for k in CRITERIA_NORM.values()}

    # Stratégie : parcourir paras, garder current_mode None|no|with, when hit header set mode,
    # when find a paragraph that seems to define a criterion (single-line contains criterion token) assign following block as context.
    current_mode = None
    for i, p in enumerate(paras):
        # detect headers
        if header_pattern_no.search(p):
            current_mode = "no-context"
            continue
        if header_pattern_with.search(p):
            current_mode = "with-context"
            continue

        # detect if paragraph mentions a criterion precisely (one of CRITERIA tokens)
        # we'll look for tokens exact (case-insensitive) - checking each criterion normalized
        tokens = [t for t in crit_keys.keys() if t in normalize_key(p)]
        if len(tokens) == 1 and current_mode is not None:
            canon = crit_keys[tokens[0]]
            # gather next paragraphs (a few) as context until we hit blank or next criterion or header
            ctx_lines = []
            j = i + 1
            # gather up to, say, 6 contiguous paras or until next header or criterion
            while j < n and len(ctx_lines) < 6:
                nxt = paras[j]
                if header_pattern_no.search(nxt) or header_pattern_with.search(nxt):
                    break
                # stop if next paragraph is another criterion headline
                nxt_norm = normalize_key(nxt)
                if any(k in nxt_norm for k in crit_keys.keys()):
                    break
                ctx_lines.append(nxt)
                j += 1
            # if we found context lines, add them; else maybe add current p as short context
            if ctx_lines:
                contexts[canon][current_mode].append("\n".join(ctx_lines))
            else:
                contexts[canon][current_mode].append(p)
    # cleanup: ensure every criterion key exists (even empty)
    for k in CRITERIA:
        contexts.setdefault(k, {"no-context": [], "with-context": []})
    print("Parsing AIMSPrompt.DOCX terminé. Critères extraits:")
    for k in contexts:
        nc = len(contexts[k]["no-context"]); wc = len(contexts[k]["with-context"])
        print(f"  {k}: no-context={nc}, with-context={wc}")
    return contexts

#--------------------------------------------------------------------------

# ==== 13) Construire un index RAG TF-IDF (corpus = contexts + phrase_pertinante sentences) ====
class SimpleRAG:
    """
    RAG léger basé sur TF-IDF pour retrieval. Indexe:
     - passages extraits depuis AIMSPrompt.DOCX (contexts)
     - phrases issues de phrase_pertinante.csv

    Fournit:
      - build_index()
      - retrieve(query, top_k)
    """

    def __init__(self):
        self.corpus = []        # list[str]
        self.metadata = []      # list[dict] same length as corpus
        self.vectorizer = None
        self.tfidf_matrix = None

    def add_passage(self, text, meta):
        self.corpus.append(text)
        self.metadata.append(meta)

    def build_index(self, max_features=50000, ngram_range=(1,2), stop_words="english"):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        print("Index RAG construit. Nombre de passages indexés:", len(self.corpus))

    def save_index(self, index_path, meta_path):
        # sauvegarde TF-IDF (matrice en sparse) et métadonnées simples
        np.savez_compressed(index_path, data=self.tfidf_matrix.data, indices=self.tfidf_matrix.indices,
                            indptr=self.tfidf_matrix.indptr, shape=self.tfidf_matrix.shape)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print("Index RAG sauvegardé:", index_path, meta_path)

    def load_index(self, index_path, meta_path):
        arr = np.load(index_path, allow_pickle=True)
        from scipy.sparse import csr_matrix
        mat = csr_matrix((arr["data"], arr["indices"], arr["indptr"]), shape=tuple(arr["shape"]))
        self.tfidf_matrix = mat
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        # recreate vectorizer not trivial; assume vectorizer is pickled separately or keep in memory.
        print("Index RAG chargé:", index_path, meta_path)

    def retrieve(self, query, top_k=5):
        """
        Retourne liste de dicts {score, text, meta} top_k les plus pertinents (cosine similarity)
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("Index non construit.")
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        idxs = sims.argsort()[::-1][:top_k]
        results = []
        for idx in idxs:
            results.append({"score": float(sims[idx]), "text": self.corpus[idx], "meta": self.metadata[idx]})
        return results

#------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# ==== 14) Construire prompts enrichis pour Mistral ====
def build_enriched_prompt(sentence, predicted_criterion, rag: SimpleRAG, contexts_dict, top_k=5):
    """
    Construit un prompt en FR qui combine:
     - instruction courte de tâche
     - phrase cible
     - contexte venant du AIMSPrompt (no-context / with-context pour le critère)
     - passages RAG (top_k)
     - consigne finale demandant au LLM Mistral de répondre si 'esclavage moderne' détecté et sur quel critère il s'appuie.
    """
    # récupérations contextuelles AIMSPrompt : both no-context and with-context for that criterion
    crit_key = predicted_criterion
    aim_no = contexts_dict.get(crit_key, {}).get("no-context", [])
    aim_with = contexts_dict.get(crit_key, {}).get("with-context", [])

    # top_k retrieved passages
    rag_hits = rag.retrieve(sentence, top_k=top_k)

    # construire prompt
    parts = []
    parts.append("Tu es un analyste expert en conformité sur le Modern Slavery Act (Australie).")
    parts.append("Tâche : à partir de la phrase fournie, décide si elle indique la présence d'un cas pertinent lié à l'esclavage moderne, et indique sur quel critère elle s'appuie (ex: Approval, Signature, Reporting entity, Structure, Operations, Supply chains, Risk description, Risk mitigation, Remediation, Effectiveness, Consultation).")
    parts.append("Réponds en français, structure ta réponse en 3 parties courtes : (1) Conclusion (Oui/Non), (2) Critère(s) utilisés et justification (2-3 phrases), (3) Sources / extraits consultés.")
    parts.append("\n---\nPhrase à analyser:\n" + sentence + "\n---\n")

    # ajouter AIMSprompt contexts si existants
    if aim_with:
        parts.append("Extrait(s) 'with-context' pour le critère '{}':".format(crit_key))
        for i, b in enumerate(aim_with[:3]):
            parts.append(f"[AIMS with-context #{i+1}]:\n{b}\n")
    if aim_no:
        parts.append("Extrait(s) 'no-context' pour le critère '{}':".format(crit_key))
        for i, b in enumerate(aim_no[:3]):
            parts.append(f"[AIMS no-context #{i+1}]:\n{b}\n")

    # ajouter passages RAG
    parts.append("Passages pertinents trouvés dans la base (RAG) :")
    for i, hit in enumerate(rag_hits):
        parts.append(f"[RAG #{i+1} score={hit['score']:.3f} source={hit['meta'].get('source')}] :\n{hit_text}".replace("{hit_text}", hit['text']))

    # instruction finale
    parts.append("\nEn te basant sur la phrase et les extraits ci-dessus, donne ta réponse (3 parties) :\n")
    return "\n\n".join(parts)

#-------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# ==== 15) Exemple d'appel à Mistral (réel + placeholder de secours) ====
import os
import requests

# Configuration — clé API et endpoint
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "uoTOdvSk0s14tC3POwuUx4X2Tw0nPKXl")
MISTRAL_MODEL = "mistral-large-latest"  # ou un autre modèle disponible
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

def call_mistral_api(prompt, max_tokens=300, temperature=0.0, top_p=1.0, timeout=30):
    """
    Appel à l'API Mistral pour génération de texte.
    
    - Utilise le modèle défini dans MISTRAL_MODEL
    - Envoie le prompt dans une structure de chat
    - Retourne le texte généré
    
    ⚠️ Assure-toi d’avoir défini la variable d’environnement MISTRAL_API_KEY
    """
    if MISTRAL_API_KEY is None or MISTRAL_API_KEY.startswith("<"):
        # Mode fallback simulation si pas de clé
        return {
            "generated_text": "SIMULATION: (Remplace call_mistral_api par ton appel API). "
                              "Interprétation : NON, la phrase ne montre pas d’esclavage moderne. "
                              "Critère : Approval (la phrase ne contient pas une approbation explicite par le conseil)."
        }

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    data = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": "Tu es un assistant expert en modern slavery compliance."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    try:
        resp = requests.post(MISTRAL_ENDPOINT, headers=headers, json=data, timeout=timeout)
    except requests.Timeout:
        return {"generated_text": f"ERREUR: Timeout ({timeout}s) dépassé lors de l'appel à Mistral."}
    except Exception as e:
        return {"generated_text": f"ERREUR: Exception lors de l'appel HTTP à Mistral: {str(e)}"}

    if resp.status_code != 200:
        return {"generated_text": f"ERREUR: API Mistral a retourné {resp.status_code} : {resp.text}"}

    result = resp.json()
    try:
        generated = result["choices"][0]["message"]["content"]
    except KeyError:
        generated = str(result)

    return {"generated_text": generated, "full_response": result}

#-----------------------------------------------------------------------------

# ==== 16) Pipeline d'assemblage RAG complet ====
def build_and_run_rag_pipeline(input_csv_path=INPUT_CSV, docx_path=AIMS_DOCX, output_phrase_csv=OUTPUT_PHRASE_CSV):
    
    # 1) créer phrase_pertinante.csv
    
    # Ici on récupère les colonnes d'intérêt exactes dans train_aims.csv
    # On suppose que les colonnes dans le csv d'origine portent les noms: Approval, Signature, c1_reporting_entity, c2_structure, c2_operation, c2_supply_chains, c3_risk_description, c4_mitigation, c4_remediation, c5_effectiveness, c6_consultation
    columns_mapping = [
        "Approval", "Signature", "c1_reporting_entity", "c2_structure", "c2_operation", "c2_supply_chains",
        "c3_risk_description", "c4_mitigation", "c4_remediation", "c5_effectiveness", "c6_consultation"
    ]
    flat = create_phrase_pertinante_csv(input_csv_path, output_phrase_csv, columns_mapping)

    # 2) parse AIMSPrompt.DOCX
    contexts = parse_aims_docx(docx_path)

    # 3) construire RAG index
    rag = SimpleRAG()

    # 3a) ajouter contexts AIMSPrompt comme passages
    for crit, d in contexts.items():
        for m in ["with-context", "no-context"]:
            for idx, txt in enumerate(d.get(m, [])):
                meta = {"source": "AIMSPrompt.DOCX", "criterion": crit, "mode": m, "index": idx}
                rag.add_passage(txt, meta)

    # 3b) ajouter phrases du CSV aplati (limiter taille pour perf si trop grand)
    SAMPLE_LIMIT = 200000  # limiter si dataset énorme
    for i, row in flat.head(SAMPLE_LIMIT).iterrows():
        txt = str(row["sentence"])
        meta = {"source": "phrase_pertinante.csv", "criterion": row["criterion"], "valeur": int(row["valeur"]), "index": int(i)}
        rag.add_passage(txt, meta)

    # 3c) construire TF-IDF index
    rag.build_index(max_features=50000, ngram_range=(1,2), stop_words="english")
    # sauvegarde index (optionnel)
    try:
        rag.save_index(str(RAG_INDEX_PATH), str(RAG_META_PATH))
    except Exception:
        pass

    return rag, contexts, flat

#-----------------------------------------------------------------------

# ==== 16) Pipeline d'assemblage RAG complet ====
def build_and_run_rag_pipeline(input_csv_path=INPUT_CSV, docx_path=AIMS_DOCX, output_phrase_csv=OUTPUT_PHRASE_CSV):
    
    # 1) créer phrase_pertinante.csv
    
    # Ici on récupère les colonnes d'intérêt exactes dans train_aims.csv
    # On suppose que les colonnes dans le csv d'origine portent les noms: Approval, Signature, c1_reporting_entity, c2_structure, c2_operation, c2_supply_chains, c3_risk_description, c4_mitigation, c4_remediation, c5_effectiveness, c6_consultation
    columns_mapping = [
        "Approval", "Signature", "c1_reporting_entity", "c2_structure", "c2_operation", "c2_supply_chains",
        "c3_risk_description", "c4_mitigation", "c4_remediation", "c5_effectiveness", "c6_consultation"
    ]
    flat = create_phrase_pertinante_csv(input_csv_path, output_phrase_csv, columns_mapping)

    # 2) parse AIMSPrompt.DOCX
    contexts = parse_aims_docx(docx_path)

    # 3) construire RAG index
    rag = SimpleRAG()

    # 3a) ajouter contexts AIMSPrompt comme passages
    for crit, d in contexts.items():
        for m in ["with-context", "no-context"]:
            for idx, txt in enumerate(d.get(m, [])):
                meta = {"source": "AIMSPrompt.DOCX", "criterion": crit, "mode": m, "index": idx}
                rag.add_passage(txt, meta)

    # 3b) ajouter phrases du CSV aplati (limiter taille pour perf si trop grand)
    SAMPLE_LIMIT = 200000  # limiter si dataset énorme
    for i, row in flat.head(SAMPLE_LIMIT).iterrows():
        txt = str(row["sentence"])
        meta = {"source": "phrase_pertinante.csv", "criterion": row["criterion"], "valeur": int(row["valeur"]), "index": int(i)}
        rag.add_passage(txt, meta)

    # 3c) construire TF-IDF index
    rag.build_index(max_features=50000, ngram_range=(1,2), stop_words="english")
    # sauvegarde index (optionnel)
    try:
        rag.save_index(str(RAG_INDEX_PATH), str(RAG_META_PATH))
    except Exception:
        pass

    return rag, contexts, flat


#--------------------------------------------------------------------------------

# ==== 17) Exemple d'utilisation : prédire / générer pour une phrase donnée ====
if __name__ == "__main__":
    # Construire l'index RAG et le CSV
    rag, contexts, flat_df = build_and_run_rag_pipeline()

    # Exemple : phrase à analyser et critère présumé (on peut aussi détecter critère via classifieur)
    example_sentence = "This statement was approved by the board on 30 June 2021 and signed by the CEO."
    
    # Si tu as un classifieur multi-label, tu peux choisir le critère le plus probable. Ici on fournit un critère d'exemple.
    predicted_criterion = "Approval"  # ou 'Signature' etc.

    prompt = build_enriched_prompt(example_sentence, predicted_criterion, rag, contexts, top_k=5)
    print("=== PROMPT ENVOYÉ À MISTRAL (extrait) ===")
    print(prompt[:1500], "...")

    # Appel réel à Mistral (à implémenter)
    resp = call_mistral_api(prompt)
    #------------------------------------------------------------
    print("=== RÉPONSE SIMULÉE DE MISTRAL ===")
    print(resp.get("generated_text"))

    # Sauvegarder exemples
    with open(DATA_DIR_OUT / "exemple/example_mistral_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    with open(DATA_DIR_OUT / "exemple/example_mistral_response.json", "w", encoding="utf-8") as f:
        json.dump(resp, f, ensure_ascii=False, indent=2)

    print("Terminé. Fichiers écrits dans", DATA_DIR_OUT)

"""

Notes opérationnelles & recommandations

* phrase_pertinante.csv : contient uniquement les lignes avec étiquettes 0 ou 1 (exclusion de -1 comme demandé). Si tu souhaites inclure -1, adapte la fonction create_phrase_pertinante_csv.

* Parsing du DOCX : j’ai utilisé une heuristique simple (détection de Prompt template (no-context) et Prompt template (with-context) et d’un critère unique sur une ligne). Selon la mise en forme réelle du DOCX, tu pourrais devoir ajuster la logique (ex. : la détection de titres, sections, ou l’association de paragraphes).

* Indexation : j’ai choisi TF-IDF (léger, explicable). En production, privilégier :

      * dense embeddings (SBERT / OpenAI embeddings) + FAISS pour retrieval plus robuste,

      * puis passage du top_k retrieved à Mistral pour la génération.

* Mistral : remplace call_mistral_api par ton client (requests ou SDK). Fournis toujours un prompt clair, limite la taille (token budget), et contrôle les réponses.

* Pipeline final : typiquement — classifier la phrase (multi-label model) → pour chaque critère prédit positif récupérer contexte via RAG → construire prompt(s) → envoyer à Mistral → agréger réponses (SHARP / règles métier) → produire un rapport.

"""
