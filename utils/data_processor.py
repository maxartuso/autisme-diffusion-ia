import pandas as pd
import numpy as np
from datetime import datetime
import re


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
        self.df = self.df.fillna({
            'description': '',
            'prix': 0,
            'categorie': 'Non classé',
            'age': 'Tous âges'
        })

        # Nettoyer les prix
        if 'prix' in self.df.columns:
            self.df['prix'] = pd.to_numeric(self.df['prix'], errors='coerce').fillna(0)

        # Créer des mots-clés si nécessaire
        if 'mots_cles' not in self.df.columns:
            self.df['mots_cles'] = self.df.apply(self.extract_keywords, axis=1)

    def extract_keywords(self, row):
        """Extrait des mots-clés d'un produit"""
        keywords = []

        # Extraire du nom
        if 'nom' in row and isinstance(row['nom'], str):
            keywords.extend(re.findall(r'\w+', row['nom'].lower()))

        # Extraire de la description
        if 'description' in row and isinstance(row['description'], str):
            keywords.extend(re.findall(r'\w+', row['description'].lower()))

        # Ajouter la catégorie
        if 'categorie' in row and isinstance(row['categorie'], str):
            keywords.append(row['categorie'].lower())

        # Filtrer les mots trop courts
        keywords = [k for k in keywords if len(k) > 2]

        return ' '.join(list(set(keywords)))

    def create_sample_catalogue(self):
        """Crée un catalogue exemple pour les tests"""
        sample_data = {
            'nom': [
                'Puzzle sensoriel arc-en-ciel',
                'Balance d\'équilibre',
                'Cartes émotions',
                'Jeu de société coopératif',
                'Casque anti-bruit enfant',
                'Balle sensorielle texturée',
                'Tableau d\'activités Montessori',
                'Livre à toucher animaux',
                'Jeu de construction magnétique',
                'Tapis de jeu interactif'
            ],
            'description': [
                'Puzzle en bois avec différentes textures pour développer le toucher',
                'Jeu d\'équilibre pour travailler la motricité globale',
                'Set de cartes illustrées pour apprendre à reconnaître les émotions',
                'Jeu de société sans compétition pour apprendre la coopération',
                'Casque réducteur de bruit adapté aux enfants hypersensibles',
                'Balle avec différentes textures pour stimulation sensorielle',
                'Tableau d\'activités pour développer la motricité fine',
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

    def export_catalogue(self, filepath):
        """Exporte le catalogue au format CSV"""
        if self.df is not None:
            self.df.to_csv(filepath, index=False)
            return True
        return False

    def get_statistics(self):
        """Retourne des statistiques sur le catalogue"""
        stats = {}

        if self.df is not None:
            stats['total_produits'] = len(self.df)

            if 'categorie' in self.df.columns:
                stats['categories'] = self.df['categorie'].value_counts().to_dict()

            if 'prix' in self.df.columns:
                stats['prix_moyen'] = self.df['prix'].mean()
                stats['prix_min'] = self.df['prix'].min()
                stats['prix_max'] = self.df['prix'].max()

            if 'disponible' in self.df.columns:
                stats['produits_disponibles'] = self.df['disponible'].sum()

        return stats