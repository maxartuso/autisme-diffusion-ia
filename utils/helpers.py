import json
import hashlib
from datetime import datetime
import re


def clean_text(text):
    """Nettoie un texte pour l'analyse"""
    if not isinstance(text, str):
        return ""

    # Convertir en minuscules
    text = text.lower()

    # Supprimer la ponctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_age_range(age_string):
    """Extrait une tranche d'âge à partir d'une chaîne"""
    if not isinstance(age_string, str):
        return None

    # Pattern pour trouver les âges
    pattern = r'(\d+)\s*[-\sà]\s*(\d+)'
    match = re.search(pattern, age_string)

    if match:
        return (int(match.group(1)), int(match.group(2)))

    # Âge unique
    pattern_single = r'(\d+)\s*ans?'
    match = re.search(pattern_single, age_string)

    if match:
        age = int(match.group(1))
        return (age, age)

    return None


def generate_product_id(name, category):
    """Génère un ID unique pour un produit"""
    base = f"{name}_{category}_{datetime.now().strftime('%Y%m')}"
    return hashlib.md5(base.encode()).hexdigest()[:8]


def format_price(price):
    """Formate un prix"""
    try:
        price_float = float(price)
        return f"{price_float:.2f} €"
    except (ValueError, TypeError):
        return "Prix non disponible"


def load_config(config_file="config.json"):
    """Charge la configuration depuis un fichier JSON"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "app_name": "Autisme Diffusion Catalogue",
            "version": "1.0.0",
            "default_language": "fr",
            "max_results": 10
        }


def save_user_preferences(user_id, preferences):
    """Sauvegarde les préférences utilisateur"""
    filename = f"data/user_preferences_{user_id}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(preferences, f, indent=2)


def load_user_preferences(user_id):
    """Charge les préférences utilisateur"""
    filename = f"data/user_preferences_{user_id}.json"
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}