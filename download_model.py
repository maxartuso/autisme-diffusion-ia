# download_model.py
import os
from sentence_transformers import SentenceTransformer


def telecharger_modele():
    """Télécharge le modèle localement pour éviter les problèmes de connexion"""
    nom_modele = "all-MiniLM-L6-v2"
    chemin_local = os.path.join("models", nom_modele)

    # Créer le dossier s'il n'existe pas
    os.makedirs(chemin_local, exist_ok=True)

    print(f"📥 Téléchargement du modèle {nom_modele}...")
    print("Cela peut prendre quelques minutes...")

    try:
        # Télécharger et sauvegarder localement
        modele = SentenceTransformer(nom_modele)
        modele.save(chemin_local)
        print(f"✅ Modèle sauvegardé dans: {chemin_local}")
        return True
    except Exception as e:
        print(f"❌ Erreur de téléchargement: {str(e)}")
        print("\n📋 Téléchargement manuel requis:")
        print(f"1. Allez sur: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2")
        print(f"2. Téléchargez tous les fichiers dans: {os.path.abspath(chemin_local)}")
        print("   Fichiers nécessaires:")
        print("   - config.json")
        print("   - pytorch_model.bin (ou model.safetensors)")
        print("   - tokenizer.json")
        print("   - tokenizer_config.json")
        print("   - vocab.txt")
        print("   - modules.json")
        return False


if __name__ == "__main__":
    telecharger_modele()