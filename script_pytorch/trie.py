import torch
from torchvision import transforms
from PIL import Image
import os
from . import cnn
import shutil
from datetime import datetime
import sys

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

def recuperer_model(num_classes, device, model_data):

    # Étape 1 – recréer le modèle avec la bonne architecture
    model = cnn.CNNIllustration(num_classes).to(device)

    # Étape 2 – charger les poids dans cette structure
    checkpoint = torch.load(model_data, map_location=device)
    model.load_state_dict(checkpoint)

    # Étape 3 – utiliser le modèle pour la prédiction
    return model.eval()

# 🔹 Prédiction
def predire_image(image_path, model, device, img_height, img_width, list_classes):
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    # Charger et prétraiter l'image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Désactiver le gradient (inference)
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Affichage
    print(f"\n🔎 Prédictions pour l'image : {os.path.basename(image_path)}")
    for idx, label in enumerate(list_classes):
        pourcentage = float(probabilities[idx]) * 100
        print(f"  ➔ {label} : {pourcentage:.2f}%")

    # Résultat final
    label_index = torch.argmax(probabilities).item()
    confidence = probabilities[label_index].item()

    return list_classes[label_index], confidence

def predire_images(dossier_images, model, device, img_height, img_width ,list_classes):
    predictions = []
    print(f"🔎 Labels connus : {list_classes}")
    # Liste des fichiers image dans le dossier
    for filename in os.listdir(dossier_images):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(dossier_images, filename)
            label, confidence = predire_image(image_path, model, device, img_height, img_width,  list_classes)
            predictions.append((filename, label, confidence))

    return predictions

def trier_images_predites(predictions, dossier_images, dossier_sortie, seuil_confiance):
    os.makedirs(dossier_sortie, exist_ok=True)

    for nom_fichier, label, confiance in predictions:
        if confiance >= seuil_confiance:
            dossier_classe = os.path.join(dossier_sortie, label)
            os.makedirs(dossier_classe, exist_ok=True)

            src = os.path.join(dossier_images, nom_fichier)
            dst = os.path.join(dossier_classe, nom_fichier)

            shutil.move(src, dst)
            print(f"✅ {nom_fichier} → {label} ({confiance:.2f})")
        else:
            print(f"❌ {nom_fichier} ignorée (confiance : {confiance:.2f})")

def ecrire_rapport(total_images_avant, images_a_classer, images_triees):

    # 🔍 Nombre total d'images après triage
    total_images_apres = len([f for f in os.listdir(images_a_classer) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # 🔹 Rapport de triage
    print("\n📌 Rapport de triage")
    print(f"Nombre total d'images avant triage : {total_images_avant}")
    print(f"Nombre d'images triées : {total_images_avant - total_images_apres}")
    print(f"Nombre d'images restantes (non triées) : {total_images_apres}\n")

    print("🔹 Détails par classe :")
    #Attention il faudra indiquer les classes derterminer par la config 
    #Ou indiquer creer automatiquement les dossier
    for dossier in os.listdir(images_triees):
        chemin_classe = os.path.join(images_triees, dossier)
        if os.path.isdir(chemin_classe):
            count = len(os.listdir(chemin_classe))
            print(f"  - {dossier} : {count} images triées")

    # Optionnel : affichage résumé
    #for nom_fichier, label, confiance in resultats:
    #    print(f"{nom_fichier} → {label} ({confiance:.2%})")