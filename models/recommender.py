import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class ProductRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = None
        self.content_similarity = None

    def prepare_features(self, df):
        """Prépare les caractéristiques pour les recommandations"""
        # Caractéristiques textuelles
        text_features = df.apply(
            lambda row: f"{row.get('nom', '')} {row.get('description', '')} {row.get('categorie', '')}", axis=1)
        self.tfidf_matrix = self.tfidf.fit_transform(text_features)

        # Similarité basée sur le contenu
        self.content_similarity = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        return self.content_similarity

    def get_recommendations(self, df, product_idx, n_recommendations=5):
        """Obtient des recommandations basées sur un produit"""
        if self.content_similarity is None:
            self.prepare_features(df)

        # Obtenir les scores de similarité
        sim_scores = list(enumerate(self.content_similarity[product_idx]))

        # Trier par similarité décroissante
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Exclure le produit lui-même
        sim_scores = sim_scores[1:n_recommendations + 1]

        # Obtenir les indices des produits recommandés
        product_indices = [i[0] for i in sim_scores]

        # Retourner les produits recommandés avec scores
        recommendations = df.iloc[product_indices].copy()
        recommendations['similarity_score'] = [i[1] for i in sim_scores]

        return recommendations

    def hybrid_recommendations(self, df, user_preferences, n_recommendations=5):
        """Recommandations hybrides basées sur les préférences utilisateur"""
        if self.content_similarity is None:
            self.prepare_features(df)

        # Créer un profil utilisateur à partir des préférences
        user_profile = self.create_user_profile(user_preferences, df)

        # Calculer les scores de similarité avec le profil utilisateur
        user_similarities = cosine_similarity(user_profile.reshape(1, -1), self.tfidf_matrix)[0]

        # Obtenir les meilleures recommandations
        top_indices = user_similarities.argsort()[-n_recommendations:][::-1]

        recommendations = df.iloc[top_indices].copy()
        recommendations['relevance_score'] = user_similarities[top_indices]

        return recommendations

    def create_user_profile(self, preferences, df):
        """Crée un profil utilisateur à partir des préférences"""
        # Cette fonction serait à adapter selon le format des préférences
        # Exemple simple: moyenne des vecteurs des produits aimés
        liked_products = preferences.get('liked_products', [])

        if liked_products:
            liked_vectors = self.tfidf_matrix[liked_products]
            user_profile = liked_vectors.mean(axis=0)
            return np.array(user_profile).flatten()
        else:
            # Retourner un vecteur moyen si pas de préférences
            return np.array(self.tfidf_matrix.mean(axis=0)).flatten()