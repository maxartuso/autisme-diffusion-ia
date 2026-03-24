import streamlit as st
import pandas as pd
import plotly.express as px
from models.recommender import ProductRecommender
from models.search_engine import SemanticSearch
from utils.data_processor import DataProcessor
import os

# Configuration de la page
st.set_page_config(
    page_title="Autisme Diffusion - Catalogue IA",
    page_icon="🧩",
    layout="wide"
)


# Initialisation des composants
@st.cache_resource
def init_components():
    data_processor = DataProcessor()
    recommender = ProductRecommender()
    search_engine = SemanticSearch()
    return data_processor, recommender, search_engine


def main():
    st.title("🧩 Autisme Diffusion - Assistant Intelligent")
    st.markdown("---")

    # Initialisation
    data_processor, recommender, search_engine = init_components()

    # Chargement des données
    df = data_processor.load_catalogue("data/catalogue.csv")

    # Sidebar - Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choisir une page",
        ["Accueil", "Recherche Intelligente", "Recommandations", "Analyse du Catalogue", "Assistant Conversationnel"]
    )

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


def show_home(df):
    st.header("📚 Bienvenue sur le Catalogue Intelligent")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Produits", len(df))
    with col2:
        st.metric("Catégories", df['categorie'].nunique() if 'categorie' in df.columns else "N/A")
    with col3:
        # Par exemple, compter les URLs non vides
        if 'url' in df.columns:
            nb_urls = df['url'].str.len().gt(0).sum()
            st.metric("URLs disponibles", nb_urls)
        else:
            st.metric("Nouveautés", len(df[df['est_nouveau'] == True]) if 'est_nouveau' in df.columns else "N/A")

    st.markdown("---")
    st.subheader("🔍 Fonctionnalités disponibles")

    st.info("""
    - **Recherche Intelligente** : Trouvez des produits avec une recherche sémantique
    - **Recommandations** : Suggestions personnalisées basées sur vos préférences
    - **Analyse du Catalogue** : Visualisations et statistiques
    - **Assistant Conversationnel** : Posez des questions en langage naturel
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
            results = search_engine.search(query, df, n_results * 2)

            # Vérifier que results n'est pas None
            if results is None:
                st.error("La recherche a retourné une erreur")
                return

            # Filtrage par catégorie si nécessaire
            if filter_category != "Toutes" and 'categorie' in results.columns and len(results) > 0:
                results = results[results['categorie'] == filter_category]

            # Limiter au nombre demandé
            results = results.head(n_results) if len(results) > 0 else results

            if len(results) > 0:
                st.success(f"✅ {len(results)} résultats trouvés")

                # Itération sécurisée sur les résultats
                for idx, row in results.iterrows():
                    # Vérifier que row n'est pas None
                    if row is None:
                        continue

                    with st.expander(f"📦 {row.get('nom', 'Produit sans nom')}"):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            # Utiliser row.get avec des valeurs par défaut
                            description = row.get('description', 'Non disponible')
                            st.write(f"**Description:** {description}")

                            categorie = row.get('categorie', 'Non spécifiée')
                            st.write(f"**Catégorie:** {categorie}")

                            # URL (optionnelle)
                            url = row.get('url', '')
                            if url and isinstance(url, str) and url.strip():
                                st.write(f"**URL:** [{url}]({url})")

                            age = row.get('age', 'Non spécifié')
                            st.write(f"**Âge recommandé:** {age}")

                        with col2:
                            # Disponibilité
                            disponible = row.get('disponible', False)
                            if disponible:
                                st.write("✅ Disponible")
                            else:
                                st.write("❌ Rupture de stock")

                            # Score de pertinence
                            score = row.get('similarity_score', None)
                            if score is not None:
                                try:
                                    st.write(f"**Pertinence:** {float(score):.2f}")
                                except (ValueError, TypeError):
                                    st.write(f"**Pertinence:** {score}")
            else:
                st.warning("Aucun résultat trouvé")


def show_recommendations(df, recommender):
    st.header("🎯 Recommandations Personnalisées")

    # Sélection du produit de référence
    product_names = df['nom'].tolist() if 'nom' in df.columns else []
    selected_product = st.selectbox("Choisissez un produit qui vous intéresse :", product_names)

    if selected_product:
        # Trouver l'index du produit
        product_idx = df[df['nom'] == selected_product].index[0]

        # Obtenir les recommandations
        recommendations = recommender.get_recommendations(df, product_idx, n_recommendations=5)

        st.subheader("Produits similaires recommandés :")

        cols = st.columns(2)
        for i, (idx, row) in enumerate(recommendations.iterrows()):
            with cols[i % 2]:
                st.markdown(f"""
                **{row['nom']}**
                - Catégorie: {row.get('categorie', 'N/A')}
                - URL: {row.get('url', 'N/A')}
                - Score de similarité: {row.get('similarity_score', 0):.0f} points
                """)
                st.markdown("---")


def show_analysis(df):
    st.header("📊 Analyse du Catalogue")

    if 'categorie' in df.columns:
        # Distribution par catégorie
        fig_categories = px.bar(
            df['categorie'].value_counts().reset_index(),
            x='categorie',
            y='count',
            title="Distribution par Catégorie"
        )
        st.plotly_chart(fig_categories, use_container_width=True)

    if 'url' in df.columns:
        st.subheader("Analyse des URLs")
        # Extraire les domaines
        df['domaine'] = df['url'].str.extract(r'https?://(?:www\.)?([^/]+)')
        domain_counts = df['domaine'].value_counts().reset_index()
        domain_counts.columns = ['domaine', 'count']
        if not domain_counts.empty:
            fig_domain = px.bar(domain_counts.head(10), x='domaine', y='count', title="Top 10 des domaines")
            st.plotly_chart(fig_domain, use_container_width=True)

    if 'age' in df.columns:
        # Analyse par âge
        fig_age = px.pie(
            df,
            names='age',
            title="Répartition par Tranche d'Âge"
        )
        st.plotly_chart(fig_age, use_container_width=True)


    # Statistiques générales (sans prix)
    st.subheader("Statistiques Générales")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total produits", len(df))
    with col2:
        st.metric("Catégories", df['categorie'].nunique() if 'categorie' in df.columns else "N/A")


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
            if 'url' in row and row['url']:
                response += f" - [Lien]({row['url']})"
            if 'description' in row:
                response += f"\n  {row['description'][:100]}...\n"
            response += "\n"

        # Statistiques si demandées
        if "combien" in query.lower() or "nombre" in query.lower():
            response += f"\n📊 Au total, nous avons {len(df)} produits dans notre catalogue."

        return response
    else:
        return "Désolé, je n'ai pas trouvé de produits correspondant à votre recherche. Pouvez-vous reformuler votre question ?"


if __name__ == "__main__":
    main()