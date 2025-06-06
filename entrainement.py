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
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 📌 Paramètres
# Chargement du fichier de configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

image_type = config['model_parameters']['image_type']
is_illustration = image_type == "illustration"

dataset_dir = config['dataset']['base_path']
aug_temp_dir = config['dataset']['augment_path']

classes_exclues = config['dataset']['classes_exclues']

# 🔹 Lister tous les dossiers du dataset sauf ceux à exclure
classes_folders = sorted([
    d for d in os.listdir(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, d)) and d not in classes_exclues
])

#classes_folders = config['dataset']['classes']
nombre_classes = len(classes_folders)
dossier_logs = config['logs']['entrainement']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path_name = config['model_parameters']['path_name']

batch_size = config['model_parameters']['batch_size']
seed = config['model_parameters']['seed']
lr= config['model_parameters']['lr']

img_height = config['model_parameters']['img_height']
img_width = config['model_parameters']['img_width']

start_time = time.time()  # Démarre le chrono

entrainement.activer_log(dossier_logs)

# 🔹 Étape 1 : Nettoyage
entrainement.nettoyer_dataset(dataset_dir, classes_folders)
# 🔹 Étape 2 : Evaluation avant augmentation
mobilenet = entrainement.charger_mobilenet(device)
entrainement.evaluer_dataset(dataset_dir, image_type, device, classes_folders, mobilenet, img_height, img_width)
# 🔹 Étape 3 : Augmentation
#entrainement.augmenter_dataset(dataset_dir, image_type, aug_temp_dir, classes_folders, seed, img_height, img_width)
# 🔹 Étape 4 : Evaluation après augmentation
#entrainement.evaluer_dataset(aug_temp_dir, image_type, device, classes_folders, mobilenet, img_height, img_width)
# 🔹 Étape 5 : Préparer le dataset
#train_loader, val_loader, test_loader = entrainement.preparer_dataset(classes_folders, aug_temp_dir, image_type, batch_size, seed, img_height, img_width)
#nombre_images = entrainement.compter_nombre_images(aug_temp_dir, classes_folders)
#epoch = entrainement.calculer_nombre_epochs(nombre_images, nombre_classes, image_type)
# 🔹 Étape 6 : Creer le modele
#model = entrainement.creer_modele(image_type, nombre_classes, device)
# 🔹 Étape 7 : Entrainer le modele
#model = entrainement.entrainer_modele(model, train_loader, val_loader, epoch, lr, device, model_path_name)
# 🔹 Étape 8 : Evaluation de l'entrainement
#entrainement.evaluer_modele(model, test_loader, nn.CrossEntropyLoss(), device, mode="Test")
#🔹 Étape 9 : Nettoyage et bilan
entrainement.supprimer_dossier_temp_aug(aug_temp_dir)

end_time = time.time()  # Arrête le chrono
train_duration = end_time - start_time
print(f"✅ Entrainement terminé en {train_duration:.2f}s")