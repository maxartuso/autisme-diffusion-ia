import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os


class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.embeddings_path = "data/embeddings/product_embeddings.pkl"

    def create_embeddings(self, df):
        """Crée des embeddings pour tous les produits"""
        # Combiner les textes pertinents
        texts = []
        for _, row in df.iterrows():
            text = f"{row.get('nom', '')} {row.get('description', '')} {row.get('categorie', '')} {row.get('mots_cles', '')}"
            texts.append(text)

        # Créer les embeddings
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        # Sauvegarder
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.embeddings, f)

        return self.embeddings

    def load_embeddings(self):
        """Charge les embeddings sauvegardés"""
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            return True
        return False

    def search(self, query, df, n_results=5):
        """Recherche sémantique"""
        # Créer ou charger les embeddings
        if self.embeddings is None:
            if not self.load_embeddings():
                self.create_embeddings(df)

        # Embedding de la requête
        query_embedding = self.model.encode([query])

        # Calculer les similarités
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Obtenir les indices des meilleurs résultats
        top_indices = similarities.argsort()[-n_results:][::-1]

        # Retourner les résultats avec scores
        results = df.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]

        return results