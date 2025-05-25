import torch
from torchvision import transforms
from PIL import Image
import os
import shutil
from datetime import datetime
import sys
import yaml

import script_pytorch.trie as trie
import time
# Chargement du fichier de configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

dossier_log = config['logs']['trie']
images_a_classer = config['images_path']['images_a_classer']
images_triees = config['images_path']['images_triees']
num_classes = len(config['dataset']['classes'])
total_images_avant = len([f for f in os.listdir(images_a_classer) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #(CPU ou GPU)

img_height = config['model_parameters']['img_height']
img_width = config['model_parameters']['img_width']
model_data = config['model_parameters']['path_name']

list_classes = config['dataset']['classes']

start_time = time.time()  # Démarre le chrono

trie.activer_log(dossier_log)
 
model = trie.recuperer_model(num_classes, device, model_data)

resultats = trie.predire_images(images_a_classer, model, device, img_height, img_width, list_classes)

trie.trier_images_predites(
    predictions=resultats,
    dossier_images=images_a_classer,
    dossier_sortie=images_triees,
    seuil_confiance=0.75
)

trie.ecrire_rapport(total_images_avant, images_a_classer, images_triees)

end_time = time.time()  # Arrête le chrono
train_duration = end_time - start_time
print(f"✅ Entrainement terminé en {train_duration:.2f}s")