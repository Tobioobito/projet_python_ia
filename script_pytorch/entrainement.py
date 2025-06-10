# Imports de la bibliothèque standard
import argparse
import copy
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime

# Imports de bibliothèques tierces
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

# Imports locaux
import script_pytorch.cnn as cnn


# 📌 Création dossier logs

def activer_log(dossier_log):
    os.makedirs(dossier_log, exist_ok=True)
    now = datetime.now().strftime('%Y_%m_%d_%H_%M')
    log_filename = os.path.join(dossier_log, f"log_{now}.txt")

    # 📌 Redirection des logs vers fichier
    class Logger:
        def __init__(self, stream, filepath):
            self.terminal = stream
            self.log = open(filepath, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()
    sys.stdout = Logger(sys.stdout, log_filename)
    return sys.stdout


def charger_mobilenet(device):

    print("CUDA disponible :", torch.cuda.is_available())
    print("Nom du GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun GPU")
    # Charger MobileNetV2 sans la dernière couche de classification
    mobilenet = models.mobilenet_v2(pretrained=True)
    mobilenet.classifier = torch.nn.Identity()
    mobilenet.eval()
    return mobilenet.to(device)

# --------------------------------------------------
# 🧹 1. Nettoyage des fichiers invalides
# --------------------------------------------------

def nettoyer_dataset(dataset_dir, classes_folders):
    err_dir = "../dataset_errone"
    os.makedirs(err_dir, exist_ok=True)

    def contient_caracteres_speciaux(texte):
        return bool(re.search(r'[^a-zA-Z0-9_\-\\/\. ]', texte))

    def extension_valide(fichier):
        return os.path.splitext(fichier)[-1].lower() in ['.jpg', '.jpeg', '.png']

    for nom in classes_folders:
            print(f"🔍 Vérification du dossier : {nom}")
            full_class_path = os.path.join(dataset_dir, nom)  # Chemin complet
            for fichier in os.listdir(full_class_path):
                chemin_complet = os.path.join(nom, fichier)
                
                if os.path.isfile(chemin_complet):
                    # 🔸 Vérification du nom de fichier et de l'extension
                    if contient_caracteres_speciaux(fichier) or not extension_valide(fichier):
                        dest = os.path.join(err_dir, nom)
                        os.makedirs(dest, exist_ok=True)
                        shutil.move(chemin_complet, os.path.join(dest, fichier))
                        print(f"⚠️ Fichier invalide déplacé : {chemin_complet}")

    print(f"⚠️ Nettoyage terminé")


# --------------------------------------------------
# 🧹 1. Evaluation
# --------------------------------------------------

def compter_references_par_classe(total_images, max_refs=10):

    percentage=0.05

    if total_images <= 0:
        return 0

    num_refs = int(total_images * percentage)
    return min(num_refs, max_refs)

def chercher_references_par_classe(class_dir, required_count):

    reference_images = []

    for fname in os.listdir(class_dir):
        if "reference" in fname.lower() and fname.lower().endswith((".jpg", ".jpeg", ".png")):
            reference_images.append(os.path.join(class_dir, fname))

    if len(reference_images) < required_count:
        raise ValueError(
            f"❌ Classe '{os.path.basename(class_dir)}' : {len(reference_images)} image(s) de référence trouvée(s), "
            f"il en faut au moins {required_count}. Ajoutez davantage d’images avec 'reference' dans le nom."
        )
    
    else:
        print(f"✅ Classe '{os.path.basename(class_dir)}' : {len(reference_images)} image(s) de référence validée(s).")
    return reference_images


def extract_feature(image_path, model, transform, device):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(img_tensor).cpu().numpy()[0]
    return feat


def evaluer_dataset(dataset_dir, is_illustration, device, classes_paths, mobilenet, img_height, img_width):
    print("\n📊 ÉVALUATION DU DATASET :", dataset_dir)

    # 🔧 Transformation de l'image pour MobileNet :
    # - Redimensionnement (Resize)
    # - Conversion en tenseur (ToTensor)
    # - Normalisation avec les valeurs d'ImageNet

    preprocess = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_images = defaultdict(list)

    for class_name in classes_paths:
        full_class_path = os.path.join(dataset_dir, class_name)
        print(f"\n🔎 Analyse du dossier : {class_name}")

        if not os.path.isdir(full_class_path):
            print(f"⚠️ Chemin non trouvé : {full_class_path}")
            continue

        all_files = [f for f in os.listdir(full_class_path)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        # Sélection des références
        required_refs = compter_references_par_classe(len(all_files))
        try:
            reference_paths = chercher_references_par_classe(full_class_path, required_refs)
        except ValueError as e:
            print(e)
            continue

        # Extraction des vecteurs de référence
        reference_features = []
        for ref_path in reference_paths:
            feat = extract_feature(ref_path, mobilenet, preprocess, device)
            reference_features.append(feat)

        reference_features = np.array(reference_features)

        # Vérification homogénéité des références
        if len(reference_features) >= 2:
            sim_matrix = cosine_similarity(reference_features)
            upper_triangle = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
            mean_ref_similarity = np.mean(upper_triangle)
            if mean_ref_similarity < 0.7:
                print(f"⚠️ Références trop hétérogènes (score : {mean_ref_similarity:.2f})")
                continue
            else:
                print(f"✅ Références valides (score : {mean_ref_similarity:.2f})")
        else:
            mean_ref_similarity = 1.0

        # Évaluation des autres images (y compris les références pour comparaison)
        for fname in all_files:
            img_path = os.path.join(full_class_path, fname)
            feat = extract_feature(img_path, mobilenet, preprocess, device)

            # Homogénéité : moyenne de similarité avec les références
            similarities = cosine_similarity([feat], reference_features)[0]
            mean_similarity = np.mean(similarities)

            # Qualité
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize((img_height, img_width))
            img_array = np.array(img_resized)
            gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            quality_score = np.std(gray) if is_illustration else cv2.Laplacian(gray, cv2.CV_64F).var()

            class_images[class_name].append((fname, mean_similarity, quality_score))

            print(f"✅ Image évaluée : {fname}")

    # Rapport final par classe
    for class_name, items in class_images.items():
        print(f"\n📂 Classe : {class_name} | Nombre d’images : {len(items)}")

        if not items:
            print("⚠️ Aucune image à évaluer.")
            continue

        items.sort(key=lambda x: x[1], reverse=True)  # tri par homogénéité décroissante

        for fname, sim, qual in items:
            sim_flag = "🔴 Mauvais" if sim < 0.5 else "🟡 Moyen" if sim < 0.7 else "🟢 Bon" if sim < 0.9 else "🔵 Excellent"
            qual_flag = "🔴 Mauvais" if qual < 30 else "🟡 Moyen" if qual < 50 else "🟢 Bon" if qual < 100 else "🔵 Excellent"
            is_reference = "⭐" if os.path.join(dataset_dir, class_name, fname) in reference_paths else " "
            print(f"🔹 {fname} {is_reference} | Homogénéité : {sim:.4f} ({sim_flag}) | Qualité : {qual:.2f} ({qual_flag})")
            #print(f"🔹 {fname} | Homogénéité : {sim:.4f} ({sim_flag}) | Qualité : {qual:.2f} ({qual_flag})")

        mean_sim = np.mean([sim for _, sim, _ in items])
        mean_qual = np.mean([qual for _, _, qual in items])
        print(f"\n📊 Moyenne homogénéité : {mean_sim:.4f}")
        print(f"📊 Moyenne qualité     : {mean_qual:.2f}")


def augmenter_dataset(base_path, image_type, augment_path, classes_folders, seed, img_height, img_width):
    print("\n🔄 AUGMENTATION DES DONNÉES...")

    # Copier tout le dataset original dans le dossier de sortie
    if os.path.exists(augment_path):
        shutil.rmtree(augment_path)  # Supprime l'ancien dossier s'il existe pour éviter les doublons

    os.makedirs(augment_path, exist_ok=True)

    # 🔹 Copier uniquement les dossiers autorisés
    for class_name in classes_folders:
        source_dir = os.path.join(base_path, class_name)
        dest_dir = os.path.join(augment_path, class_name)

        if os.path.exists(source_dir):
            shutil.copytree(source_dir, dest_dir)
        else:
            print(f"⚠️ Dossier introuvable : {source_dir}")

    random.seed(seed)
    torch.manual_seed(seed)

    # 1. Définir les transformations
    if image_type == "illustration":
        augmentation = transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.RandomResizedCrop(size=img_height, scale=(0.9, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.1)], p=0.7),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])
    else:
        augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(size=img_height, scale=(0.9, 1.0)),
            transforms.ColorJitter(contrast=0.1)
        ])

    # 2. Rassembler les images par classe
    class_images = defaultdict(list)

    for class_name in classes_folders:
        class_path = os.path.join(augment_path, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"🔎 Traitement de la classe : {class_name}")

        for fname in os.listdir(class_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(class_path, fname)
                class_images[class_name].append(full_path)

    # 3. Calculer combien d’images ajouter par classe
    counts = {label: len(paths) for label, paths in class_images.items()}
    max_count = max(counts.values())

    for label, image_paths in class_images.items():
        current_count = len(image_paths)
        extra_needed = max(1, int(current_count * 0.05)) if current_count == max_count else max_count - current_count

        print(f"🔧 {label} → Ajout de {extra_needed} images")
        save_dir = os.path.join(augment_path, label)
        os.makedirs(save_dir, exist_ok=True)

        # 4. Générer des images augmentées
        for i in tqdm(range(extra_needed), desc=f""):
            path = random.choice(image_paths)
            image = Image.open(path).convert("RGB").resize((img_height, img_width))
            aug_image = augmentation(image)
            aug_image.save(os.path.join(save_dir, f"aug_{i}.jpg"))


def preparer_dataset(classes_autorisees, root_dir, type_images, batch_size, seed, img_height, img_width):

    #train_loader : pour l'apprentissage. Le modèle apprend à classer. Apprendre avec des exercices. données vues par le modèle pendant l'apprentissage.
    #val_loader : évaluation pendant l'entraînement, souvent pour détecter l’overfitting. . pour surveiller la convergence / overfitting. Vérifie les progrès à chaque époque d’entraînement. Faire des exercices similaires pour voir si tu progresses.
    #test_loader : données jamais vues par le modèle, utilisées à la toute fin pour l’évaluation finale. .pour évaluer la généralisation. Mesure la vraie performance finale.  Prouver que tu maîtrises vraiment, avec des questions nouvelles.

    print(f"\n🔄 Préparation du dataset : {root_dir}")
    print(f"🔎 Classes autorisées : {classes_autorisees}")
    
    torch.manual_seed(seed)

    # Transformation en fonction du type d'image
    if type_images == "illustration":
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
        ])
    else:  # photo
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    
    # 🔹 Vérifier uniquement les dossiers autorisés
    dossiers_a_garder = [d for d in os.listdir(root_dir) if d in classes_autorisees]
    if not dossiers_a_garder:
        raise ValueError("❌ Aucune classe valide trouvée dans le dataset. Vérifiez le fichier de config.")
    
    # 🔹 Créer un dataset temporaire pour charger les images
    full_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    
    total_images = len(full_dataset)
    print(full_dataset.class_to_idx)
    print(total_images)
    # Calculer les longueurs de chaque split
    train_len = int(0.7 * total_images)
    val_len = int(0.15 * total_images)
    test_len = total_images - train_len - val_len

    # Diviser le dataset
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_len, val_len, test_len])

    # Créer les DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader

# --- Fonction de création ---
def creer_modele(image_type: str, num_classes: int, device):
    print(f"🧠 Création modèle : type={image_type}")
    if image_type == "illustration":
        model = cnn.CNNIllustration(num_classes)
    else:
        model = cnn.CNNPhoto(num_classes)

    print("CUDA disponible :", torch.cuda.is_available())
    print("Nom du GPU :", device)
    return model.to(device)

def compter_nombre_images(aug_temp_dir, classes_folders):
    total_images = 0

    for classe in classes_folders:
        dossier_classe = os.path.join(aug_temp_dir, classe)
        total_images += len(os.listdir(dossier_classe))

    return total_images

"""
def compter_nombre_images(aug_temp_dir):
    total_images = 0

    for dossier in os.listdir(aug_temp_dir):
        chemin_dossier = os.path.join(aug_temp_dir, dossier)
        if os.path.isdir(chemin_dossier):
            total_images += len(os.listdir(chemin_dossier))

    return total_images
"""


def calculer_nombre_epochs(nombre_images, nombre_classes, type_image):
 
    base_images = 1000        # Référence : 1000 images
    base_classes = 10         # Référence : 10 classes
    base_epochs = 30          # 30 époques pour la base

    # Facteurs multiplicateurs
    facteur_images = nombre_images / base_images
    facteur_classes = (nombre_classes / base_classes) ** 0.5  # racine carrée : évite la croissance explosive
    facteur_type = 1.3 if type_image == "illustration" else 1.0

    epochs = int(base_epochs * facteur_images * facteur_classes * facteur_type)

    # Plafond et plancher pour éviter les extrêmes
    epochs = max(10, min(epochs, 500))

    return epochs

def entrainer_modele(model, train_loader, val_loader, epochs, learning_rate, device, model_path_name):

    #val_loss = À quel point le modèle est sûr de ses prédictions (même s’il se trompe).
    #val_accuracy = À quel point le modèle a raison (peu importe la confiance dans sa prédiction).

    #val_loss est une valeur positive flottante (souvent 0.1 – 2.0, mais peut être bien plus si le modèle est instable).
    #val_acc est entre 0 et 1 (0–100% exprimé en ratio).

    print("CUDA disponible :", torch.cuda.is_available())
    print("Nom du GPU :", device)
    model.to(device)
    #patience = round(epochs/4)
    patience = 10
    poids = 2.0  # 🔧 Poids de l'accuracy dans le score global (ajustable)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    meilleur_modele = copy.deepcopy(model.state_dict())
    meilleure_val_loss = float('inf')
    epochs_sans_amélioration = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        print(f"\n🔁 Époque {epoch+1}/{epochs}")
        model.train()
        total_train_loss = 0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc="Entraînement"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            #print("Sortie modèle :", outputs[0])
            #print("Sortie modèle :", outputs[:2])
            #print(f"\nLabels batch :", labels)
            #print(f"\nMax label :", labels.max().item())
            #print(f"\nShape sortie modèle :", outputs.shape)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        val_loss, val_acc = evaluer_modele(model, val_loader, criterion, device, mode="Validation")

        train_losses.append(total_train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        score = val_loss - (val_acc * poids) # Calcul du score combiné (plus bas = mieux)

        print(f"✅ Train loss: {total_train_loss:.4f} | Train acc: {train_acc:.2%}")
        print(f"🧪 Val loss: {val_loss:.4f} | Val acc: {val_acc:.2%}")
        print(f"🧮 Score : {score:.2f}")

        if score < meilleure_val_loss:
            meilleure_val_loss = score
            meilleur_modele = copy.deepcopy(model.state_dict())
            meilleure_epoque = epoch + 1  # +1 car les époques sont 0-indexées
            meilleure_val_acc = val_acc
            meilleure_val_loss_brute = val_loss
            meilleure_val_score = score
            epochs_sans_amélioration = 0
            print(f"💾 Meilleur modèle sauvegardé.")
                #f"(Val Loss : {meilleure_val_loss_brute:.4f} | "
                #f"Val Acc : {meilleure_val_acc:.2%} | "
                #f"Score : {meilleure_val_score:.4f})")

        else:
            epochs_sans_amélioration += 1
            print(f"⏳ Pas d'amélioration ({epochs_sans_amélioration}/{patience})")

            if epochs_sans_amélioration >= patience:
                print("🛑 Early stopping déclenché.")
                break

    # Charger le meilleur modèle
    model.load_state_dict(meilleur_modele)
    torch.save(model.state_dict(), model_path_name)
    print(f"\n🏁 Entraînement terminé. Meilleur modèle à l'époque {meilleure_epoque} "
      f"(Val Loss : {meilleure_val_loss_brute:.4f} | "
      f"Val Acc : {meilleure_val_acc:.2%} | "
      f"Score : {meilleure_val_score:.4f})")

    # Courbes
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Courbe de Loss")
    plt.xlabel("Époque")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Acc")
    plt.plot(val_accuracies, label="Val Acc")
    plt.title("Courbe de Précision")
    plt.xlabel("Époque")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model

def evaluer_modele(model, dataloader, criterion, device, mode="Test"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"\n📊 [{mode}] Loss : {avg_loss:.4f} | Accuracy : {accuracy:.2%}")
    return avg_loss, accuracy

def supprimer_dossier_temp_aug(aug_temp_dir):

    if os.path.exists(aug_temp_dir):
        shutil.rmtree(aug_temp_dir)
        print(f"🧹 Dossier temporaire supprimé : {aug_temp_dir}")