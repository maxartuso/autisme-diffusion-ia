#!/bin/bash

# Installation des dépendances
echo "Installation des dépendances..."
pip install -r requirements.txt

# Création des dossiers nécessaires
mkdir -p data/embeddings
mkdir -p models
mkdir -p utils
mkdir -p templates

# Lancement de l'application
echo "Lancement de l'application..."
streamlit run app.py