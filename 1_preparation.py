# Imports de la bibliothèque standard
import os
import time

# Imports de bibliothèques tierces
import torch
import yaml

from utils.logger import activer_log

# Imports locaux
import script_pytorch.preparation as preparation

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
dossier_logs = config['logs']['preparation']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path_name = config['model_parameters']['path_name']

batch_size = config['model_parameters']['batch_size']
seed = config['model_parameters']['seed']

img_height = config['model_parameters']['img_height']
img_width = config['model_parameters']['img_width']

start_time = time.time()  # Démarre le chrono

activer_log(dossier_logs)

# 🔹 Étape 1 : Nettoyage
preparation.nettoyer_dataset(dataset_dir, classes_folders)
# 🔹 Étape 2 : Evaluation avant augmentation
mobilenet = preparation.charger_mobilenet(device)
classement_images = preparation.evaluer_dataset(dataset_dir, image_type, device, classes_folders, mobilenet, img_height, img_width)

# 🔹 Étape 3 : Augmentation
preparation.augmenter_dataset(dataset_dir, image_type, aug_temp_dir, classes_folders, seed, img_height, img_width, classement_images)
# 🔹 Étape 4 : Evaluation après augmentation
preparation.evaluer_dataset(aug_temp_dir, image_type, device, classes_folders, mobilenet, img_height, img_width)