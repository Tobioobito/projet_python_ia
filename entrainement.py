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

import script_pytorch.entrainement as entrainement

# log nettoyage automatique

# 📌 Paramètres
# Chargement du fichier de configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

image_type = config['model_parameters']['image_type']
is_illustration = image_type == "illustration"

dataset_dir = config['dataset']['base_path']
aug_temp_dir = config['dataset']['augment_path']
classes_folders = config['dataset']['classes']
nombre_classes = len(classes_folders)
dossier_logs = config['logs']['entrainement']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path_name = config['model_parameters']['path_name']


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

batch_size = 32
seed = 42
lr=1e-4
start_time = time.time()  # Démarre le chrono

entrainement.activer_log(dossier_logs)

# 🔹 Étape 1 : Nettoyage
entrainement.nettoyer_dataset(dataset_dir, classes_folders)
# 🔹 Étape 2 : Evaluation avant augmentation
mobilenet = entrainement.charger_mobilenet(device)
entrainement.evaluer_dataset(dataset_dir, image_type, device, classes_folders, mobilenet)
# 🔹 Étape 3 : Augmentation
entrainement.augmenter_dataset(dataset_dir, image_type, aug_temp_dir, classes_folders)
# 🔹 Étape 4 : Evaluation après augmentation
entrainement.evaluer_dataset(aug_temp_dir, image_type, device, classes_folders, mobilenet)
# 🔹 Étape 5 : Préparer le dataset
train_loader, val_loader, test_loader = entrainement.preparer_dataset(classes_folders, aug_temp_dir, image_type, batch_size, seed)
# 🔹 Étape 6 : Creer le modele
model = entrainement.creer_modele(image_type, nombre_classes, device)
# 🔹 Étape 7 : Entrainer le modele
nombre_images = compter_nombre_images(aug_temp_dir, classes_folders)
epoch = calculer_nombre_epochs(nombre_images, nombre_classes, image_type)
model = entrainement.entrainer_modele(model, train_loader, val_loader, epoch, lr, device, model_path_name)
# 🔹 Étape 8 : Evaluation de l'entrainement
entrainement.evaluer_modele(model, test_loader, nn.CrossEntropyLoss(), device)
#🔹 Étape 9 : Nettoyage et bilan
entrainement.supprimer_dossier_temp_aug(aug_temp_dir)

end_time = time.time()  # Arrête le chrono
train_duration = end_time - start_time
print(f"✅ Entrainement terminé en {train_duration:.2f}s")