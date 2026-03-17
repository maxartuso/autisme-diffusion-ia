import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import os

# Configuration de la page
st.set_page_config(
    page_title="Autisme Diffusion - Catalogue IA",
    page_icon="🧩",
    layout="wide"
)


# ==================== CLASSES SANS SCIKIT-LEARN ====================

class DataProcessor:
    def __init__(self):
        self.df = None

    def load_catalogue(self, filepath):
        """Charge le catalogue depuis un fichier CSV"""
        try:
            self.df = pd.read_csv(filepath)
            self.clean_data()
            return self.df
        except FileNotFoundError:
            # Créer un catalogue exemple si le fichier n'existe pas
            return self.create_sample_catalogue()

    def clean_data(self):
        """Nettoie et prépare les données"""
        # Nettoyer les noms de colonnes
        self.df.columns = [col.strip().lower().replace(' ', '_') for col in self.df.columns]

        # Gérer les valeurs manquantes
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].fillna('')

        # Nettoyer les prix
        if 'prix' in self.df.columns:
            self.df['prix'] = pd.to_numeric(self.df['prix'], errors='coerce').fillna(0)

    def create_sample_catalogue(self):
        """Crée un catalogue exemple pour les tests"""
        sample_data = {
            'nom': [
                'Puzzle sensoriel arc-en-ciel',
                "Balance d'équilibre",
                'Cartes émotions',
                'Jeu de société coopératif',
                'Casque anti-bruit enfant',
                'Balle sensorielle texturée',
                "Tableau d'activités Montessori",
                'Livre à toucher animaux',
                'Jeu de construction magnétique',
                'Tapis de jeu interactif'
            ],
            'description': [
                'Puzzle en bois avec différentes textures pour développer le toucher',
                "Jeu d'équilibre pour travailler la motricité globale",
                'Set de cartes illustrées pour apprendre à reconnaître les émotions',
                'Jeu de société sans compétition pour apprendre la coopération',
                'Casque réducteur de bruit adapté aux enfants hypersensibles',
                'Balle avec différentes textures pour stimulation sensorielle',
                "Tableau d'activités pour développer la motricité fine",
                'Livre avec différentes matières pour éveil sensoriel',
                'Jeu de construction avec aimants pour créativité',
                'Tapis avec zones interactives pour jeux d\'éveil'
            ],
            'categorie': [
                'Motricité fine',
                'Motricité globale',
                'Compétences sociales',
                'Jeux',
                'Sensorialité',
                'Sensorialité',
                'Motricité fine',
                'Éveil',
                'Créativité',
                'Éveil'
            ],
            'prix': [24.99, 34.50, 12.90, 29.99, 39.90, 8.50, 45.00, 15.90, 32.00, 55.00],
            'age': ['3-6 ans', '4-8 ans', '4-10 ans', '5-12 ans', '3-12 ans', '0-3 ans', '2-5 ans', '1-3 ans',
                    '4-10 ans', '6 mois-3 ans'],
            'disponible': [True, True, False, True, True, True, False, True, True, True],
            'est_nouveau': [True, False, True, False, True, False, False, True, False, True]
        }

        self.df = pd.DataFrame(sample_data)
        return self.df


class SemanticSearch:
    def __init__(self):
        pass

    def search(self, query, df, n_results=5):
        """Recherche basée sur des mots-clés (sans sklearn)"""
        query = query.lower()
        query_words = set(query.split())
        scores = []

        for idx, row in df.iterrows():
            score = 0
            # Combiner tous les textes pertinents
            text = f"{row.get('nom', '')} {row.get('description', '')} {row.get('categorie', '')}".lower()

            # Compter les correspondances de mots
            for word in query_words:
                if word in text:
                    score += 1
                    # Bonus si le mot est dans le titre
                    if word in row.get('nom', '').lower():
                        score += 2

            scores.append(score)

        # Créer un DataFrame avec les scores
        results_df = df.copy()
        results_df['similarity_score'] = scores

        # Filtrer les résultats avec score > 0 et trier
        results_df = results_df[results_df['similarity_score'] > 0]
        results_df = results_df.sort_values('similarity_score', ascending=False).head(n_results)

        return results_df


class ProductRecommender:
    def __init__(self):
        pass

    def get_recommendations(self, df, product_idx, n_recommendations=5):
        """Recommandations basées sur la catégorie et les mots-clés"""
        product = df.iloc[product_idx]
        product_category = product.get('categorie', '')
        product_name = product.get('nom', '')

        # Extraire les mots-clés du produit
        product_words = set(str(product_name).lower().split())

        scores = []
        for idx, row in df.iterrows():
            if idx == product_idx:
                scores.append(0)
                continue

            score = 0
            # Similarité de catégorie
            if row.get('categorie', '') == product_category:
                score += 10

            # Similarité de mots dans le nom
            row_words = set(str(row.get('nom', '')).lower().split())
            common_words = product_words.intersection(row_words)
            score += len(common_words) * 2

            scores.append(score)

        # Obtenir les indices des meilleurs scores
        top_indices = np.argsort(scores)[-n_recommendations:][::-1]

        # Retourner les résultats
        results = df.iloc[top_indices].copy()
        results['similarity_score'] = [scores[i] for i in top_indices]

        return results


# ==================== FONCTIONS DE L'INTERFACE ====================

def show_home(df):
    st.header("📚 Bienvenue sur le Catalogue Intelligent")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Produits", len(df))
    with col2:
        st.metric("Catégories", df['categorie'].nunique() if 'categorie' in df.columns else "N/A")
    with col3:
        st.metric("Nouveautés", len(df[df['est_nouveau'] == True]) if 'est_nouveau' in df.columns else "N/A")

    st.markdown("---")
    st.subheader("🔍 Fonctionnalités disponibles")

    st.info("""
    - **Recherche Intelligente** : Trouvez des produits par mots-clés
    - **Recommandations** : Suggestions basées sur les catégories
    - **Analyse du Catalogue** : Visualisations et statistiques
    - **Assistant Conversationnel** : Posez des questions simples
    """)

    # Aperçu du catalogue
    st.subheader("📋 Aperçu du Catalogue")
    st.dataframe(df.head(10), use_container_width=True)


def show_search(df, search_engine):
    st.header("🔎 Recherche Intelligente")

    query = st.text_input("Que recherchez-vous ?", placeholder="Ex: jeux pour développer la motricité fine...")

    col1, col2 = st.columns(2)
    with col1:
        n_results = st.slider("Nombre de résultats", 1, 20, 5)
    with col2:
        categories = ["Toutes"] + list(df['categorie'].unique()) if 'categorie' in df.columns else ["Toutes"]
        filter_category = st.selectbox("Filtrer par catégorie", categories)

    if query:
        with st.spinner("Recherche en cours..."):
            # Recherche
            results = search_engine.search(query, df, n_results * 2)  # On cherche plus pour ensuite filtrer

            # Filtrage par catégorie si nécessaire
            if filter_category != "Toutes" and 'categorie' in results.columns:
                results = results[results['categorie'] == filter_category]

            # Limiter au nombre demandé
            results = results.head(n_results)

            if len(results) > 0:
                st.success(f"✅ {len(results)} résultats trouvés")

                for idx, row in results.iterrows():
                    with st.expander(f"📦 {row['nom']}"):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(f"**Description:** {row.get('description', 'Non disponible')}")
                            st.write(f"**Catégorie:** {row.get('categorie', 'Non spécifiée')}")
                            st.write(f"**Prix:** {row.get('prix', 'Non disponible')} €")
                            st.write(f"**Âge recommandé:** {row.get('age', 'Non spécifié')}")
                        with col2:
                            if 'disponible' in row:
                                status = "✅ Disponible" if row['disponible'] else "❌ Rupture de stock"
                                st.write(status)
                            if 'similarity_score' in row:
                                st.write(f"**Pertinence:** {row['similarity_score']:.0f} points")
            else:
                st.warning("Aucun résultat trouvé")


def show_recommendations(df, recommender):
    st.header("🎯 Recommandations Personnalisées")

    # Sélection du produit de référence
    if 'nom' in df.columns:
        product_names = df['nom'].tolist()
        selected_product = st.selectbox("Choisissez un produit qui vous intéresse :", product_names)

        if selected_product:
            # Trouver l'index du produit
            product_idx = df[df['nom'] == selected_product].index[0]

            # Obtenir les recommandations
            recommendations = recommender.get_recommendations(df, product_idx, n_recommendations=5)

            st.subheader("Produits similaires recommandés :")

            for idx, row in recommendations.iterrows():
                st.markdown(f"""
                **{row['nom']}**
                - Catégorie: {row.get('categorie', 'N/A')}
                - Prix: {row.get('prix', 'N/A')} €
                - Score de similarité: {row.get('similarity_score', 0):.0f} points
                """)
                st.markdown("---")


def show_analysis(df):
    st.header("📊 Analyse du Catalogue")

    if 'categorie' in df.columns:
        # Distribution par catégorie
        cat_counts = df['categorie'].value_counts().reset_index()
        cat_counts.columns = ['categorie', 'count']
        fig_categories = px.bar(
            cat_counts,
            x='categorie',
            y='count',
            title="Distribution par Catégorie"
        )
        st.plotly_chart(fig_categories, use_container_width=True)

    if 'prix' in df.columns:
        # Distribution des prix
        fig_prix = px.histogram(
            df,
            x='prix',
            nbins=30,
            title="Distribution des Prix"
        )
        st.plotly_chart(fig_prix, use_container_width=True)

    if 'age' in df.columns:
        # Analyse par âge
        age_counts = df['age'].value_counts().reset_index()
        age_counts.columns = ['age', 'count']
        fig_age = px.pie(
            age_counts,
            names='age',
            values='count',
            title="Répartition par Tranche d'Âge"
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # Statistiques détaillées
    st.subheader("Statistiques Détaillées")
    if 'prix' in df.columns:
        stats = df['prix'].describe()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Prix Moyen", f"{stats['mean']:.2f} €")
        with col2:
            st.metric("Prix Minimum", f"{stats['min']:.2f} €")
        with col3:
            st.metric("Prix Maximum", f"{stats['max']:.2f} €")
        with col4:
            st.metric("Médiane", f"{stats['50%']:.2f} €")


def show_chatbot(df, search_engine):
    st.header("💬 Assistant Conversationnel")

    # Initialiser l'historique
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Afficher l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Zone de saisie
    if prompt := st.chat_input("Posez votre question sur le catalogue..."):
        # Afficher le message de l'utilisateur
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Générer la réponse
        with st.chat_message("assistant"):
            with st.spinner("Réflexion en cours..."):
                response = generate_chat_response(prompt, df, search_engine)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


def generate_chat_response(query, df, search_engine):
    """Génère une réponse contextuelle basée sur la requête"""

    # Recherche de produits pertinents
    results = search_engine.search(query, df, n_results=3)

    if len(results) > 0:
        response = f"J'ai trouvé des produits qui pourraient vous intéresser :\n\n"
        for idx, row in results.iterrows():
            response += f"• **{row['nom']}**"
            if 'prix' in row:
                response += f" - {row['prix']} €"
            if 'description' in row:
                response += f"\n  {row['description'][:100]}...\n"
            response += "\n"

        # Statistiques si demandées
        if "combien" in query.lower() or "nombre" in query.lower():
            response += f"\n📊 Au total, nous avons {len(df)} produits dans notre catalogue."

        return response
    else:
        return "Désolé, je n'ai pas trouvé de produits correspondant à votre recherche. Pouvez-vous reformuler votre question ?"


# ==================== APPLICATION PRINCIPALE ====================

def main():
    st.title("🧩 Autisme Diffusion - Assistant Intelligent")
    st.markdown("---")

    # Initialisation
    data_processor = DataProcessor()
    recommender = ProductRecommender()
    search_engine = SemanticSearch()

    # Chargement des données
    df = data_processor.load_catalogue("data/catalogue.csv")

    # Sidebar - Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choisir une page",
        ["Accueil", "Recherche Intelligente", "Recommandations", "Analyse du Catalogue", "Assistant Conversationnel"]
    )

    # Affichage de la page sélectionnée
    if page == "Accueil":
        show_home(df)
    elif page == "Recherche Intelligente":
        show_search(df, search_engine)
    elif page == "Recommandations":
        show_recommendations(df, recommender)
    elif page == "Analyse du Catalogue":
        show_analysis(df)
    elif page == "Assistant Conversationnel":
        show_chatbot(df, search_engine)


if __name__ == "__main__":
    main()