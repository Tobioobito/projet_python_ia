# install.py

import os
import subprocess
import sys

# Liste des dossiers à créer
folders = [
    "dataset_init/classe_1",
    "dataset_init/classe_2",
    "dataset_init/classe_3",
    "images_a_classer",
    "images_triees"
    "logs/entrainement"
    "logs/trie"
]

# Chemin vers le fichier requirements.txt
requirements_file = "requirements.txt"

def create_folders(folder_list):
    for folder in folder_list:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"[OK] Dossier créé : {folder}")
        except Exception as e:
            print(f"[ERREUR] Création de {folder} : {e}")

def install_requirements(file_path):
    if not os.path.exists(file_path):
        print(f"[ERREUR] Fichier non trouvé : {file_path}")
        return

    try:
        print(f"[...] Installation des dépendances depuis {file_path}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", file_path])
        print("[OK] Installation terminée.")
    except subprocess.CalledProcessError:
        print("[ERREUR] Échec de l'installation des dépendances.")

if __name__ == "__main__":
    print(">>> Création des dossiers...")
    create_folders(folders)

    print("\n>>> Installation des packages depuis requirements.txt...")
    install_requirements(requirements_file)

    print("\n✅ Installation terminée.")
