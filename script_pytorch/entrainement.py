import os
from datetime import datetime

import sys
import time
import re

import cv2
import torch
import numpy as np
from torchvision import models, transforms, datasets
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from PIL import Image
import shutil

import random
from tqdm import tqdm

import argparse

import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import torch.optim as optim
import copy

import yaml
import script_pytorch.cnn as cnn

import matplotlib.pyplot as plt

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

def evaluer_dataset(dataset_dir, is_illustration, device, classes_paths, mobilenet):
    """
    Évalue les images du dataset en extrayant les features et en calculant la similarité et la qualité.

    Args:
    - config (dict): Configuration YAML contenant les chemins des datasets.
    - is_illustration (bool): True si c'est un dataset d'illustration, False pour des photos.
    - device (torch.device): L'appareil utilisé (CPU ou GPU).
    - mobilenet (torch.nn.Module): Le modèle MobileNet pour l'extraction de features.
    """
    
    # 🔹 Extraction des chemins à partir du fichier de config
    
    print("\n📊 ÉVALUATION DU DATASET :", dataset_dir)

    # 🔹 Transformation des images pour MobileNet
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_images = defaultdict(list)

    # 🔹 Extraction des features par classe
    for class_name in classes_paths:
        full_class_path = os.path.join(dataset_dir, class_name)  # Chemin complet
        print(f"🔎 Analyse du dossier : {class_name}")
        
        if not os.path.isdir(full_class_path):
            print(f"⚠️ Chemin non trouvé : {full_class_path}")
            continue

        for fname in os.listdir(full_class_path):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(full_class_path, fname)
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = mobilenet(img_tensor).cpu().numpy()[0]

            img_array = np.array(img.resize((224, 224)))
            class_images[class_name].append((features, img_array))

    # 🔹 Rapport de l'évaluation
    print("\n🔎 Rapport :")
    for class_name, items in class_images.items():
        embeddings = [feat for feat, _ in items]
        images = [img for _, img in items]

        if len(embeddings) >= 2:
            sim_matrix = cosine_similarity(embeddings)
            upper_triangle = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
            mean_sim = np.mean(upper_triangle)
        else:
            mean_sim = 1.0

        quality_scores = []
        for img in images:
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            score = np.std(gray) if is_illustration else cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_scores.append(score)

        print(f"🗂️ {class_name} : {len(images)} images | Homogénéité = {mean_sim:.2f} | Qualité = {np.mean(quality_scores):.2f}")


def augmenter_dataset(base_path, image_type, augment_path, classes_folders, seed=42):
    print("\n🔄 AUGMENTATION DES DONNÉES...")

    # Copier tout le dataset original dans le dossier de sortie
    if os.path.exists(augment_path):
        shutil.rmtree(augment_path)  # Supprime l'ancien dossier s'il existe pour éviter les doublons

    shutil.copytree(base_path, augment_path)

    random.seed(seed)
    torch.manual_seed(seed)

    # 1. Définir les transformations
    if image_type == "illustration":
        augmentation = transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.1)], p=0.7),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])
    else:
        augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.ColorJitter(contrast=0.1)
        ])

    # 2. Rassembler les images par classe
    class_images = defaultdict(list)

    for class_name in classes_folders:
            full_class_path = os.path.join(base_path, class_name)  # Chemin complet

            if not os.path.isdir(full_class_path):
                print(f"⚠️ Chemin non trouvé : {class_name}")
                continue

            print(f"🔎 Traitement de la classe : {class_name}")
            output_class_path = os.path.join(augment_path, class_name)
            os.makedirs(output_class_path, exist_ok=True)

            for fname in os.listdir(output_class_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    original_path = os.path.join(full_class_path, fname)
                    
                    shutil.copy(original_path, output_class_path)  # Copie de l'image dans le dossier de sortie
                    class_images[class_name].append(original_path)

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
            image = Image.open(path).convert("RGB").resize((224, 224))
            aug_image = augmentation(image)
            aug_image.save(os.path.join(save_dir, f"aug_{i}.jpg"))

def preparer_dataset(classes_autorisees, root_dir, type_images, batch_size, seed):

    print(f"\n🔄 Préparation du dataset : {root_dir}")
    print(f"🔎 Classes autorisées : {classes_autorisees}")
    
    torch.manual_seed(seed)

    # Transformation en fonction du type d'image
    if type_images == "illustration":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    else:  # photo
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    
    # 🔹 Vérifier uniquement les dossiers autorisés
    dossiers_a_garder = [d for d in os.listdir(root_dir) if d in classes_autorisees]
    if not dossiers_a_garder:
        raise ValueError("❌ Aucune classe valide trouvée dans le dataset. Vérifiez le fichier de config.")
    
    # 🔹 Construire un mapping restreint
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(dossiers_a_garder)}
    print(class_to_idx)
    # 🔹 Créer un dataset temporaire pour charger les images
    full_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    
    # 🔹 Filtrer les données en fonction du mapping
    samples_filtres = []
    for image_path, _ in full_dataset.samples:
        class_name = os.path.basename(os.path.dirname(image_path))
        if class_name in class_to_idx:
            samples_filtres.append((image_path, class_to_idx[class_name]))

    # 🔹 Appliquer les samples filtrés au dataset
    full_dataset.samples = samples_filtres
    print(full_dataset.samples)
    full_dataset.class_to_idx = class_to_idx
    print(full_dataset.class_to_idx)
    total_images = len(full_dataset)
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



def entrainer_modele(model, train_loader, val_loader, epochs, lr, device, model_path_name):

    print("CUDA disponible :", torch.cuda.is_available())
    print("Nom du GPU :", device)
    model.to(device)
    #patience = round(epochs/4)
    patience = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        val_loss, val_acc = evaluer_modele(model, val_loader, criterion, device)

        train_losses.append(total_train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"✅ Train loss: {total_train_loss:.4f} | Train acc: {train_acc:.2%}")
        print(f"🧪 Val loss: {val_loss:.4f} | Val acc: {val_acc:.2%}")

        # Early stopping
        if val_loss < meilleure_val_loss:
            meilleure_val_loss = val_loss
            meilleur_modele = copy.deepcopy(model.state_dict())
            epochs_sans_amélioration = 0
            print("💾 Meilleur modèle sauvegardé !")
        else:
            epochs_sans_amélioration += 1
            print(f"⏳ Pas d'amélioration ({epochs_sans_amélioration}/{patience})")

            if epochs_sans_amélioration >= patience:
                print("🛑 Early stopping déclenché.")
                break

    # Charger le meilleur modèle
    model.load_state_dict(meilleur_modele)
    torch.save(model.state_dict(), model_path_name)
    print("💾 Modèle sauvegardé dans :", model_path_name)

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

def evaluer_modele(model, dataloader, criterion, device):
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
    print(f"\n📊 Test Loss : {avg_loss:.4f} | Test Accuracy : {accuracy:.2%}")
    return avg_loss, accuracy

def supprimer_dossier_temp_aug(aug_temp_dir):

    if os.path.exists(aug_temp_dir):
        shutil.rmtree(aug_temp_dir)
        print(f"🧹 Dossier temporaire supprimé : {aug_temp_dir}")