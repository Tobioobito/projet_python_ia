# Imports de la bibliothèque standard
import os
import random
import re
import shutil
from collections import defaultdict

# Imports de bibliothèques tierces
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from tqdm import tqdm

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

    print("⚠️ Nettoyage terminé")


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
        print(f"\n📊 Évaluation des images pour la classe : {class_name}")
        for fname in tqdm(all_files, desc="🔍 Évaluation"):
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

    sorted_images_by_class = {}

    # Rapport final par classe
    for class_name, items in class_images.items():
        print(f"\n📂 Classe : {class_name} | Nombre d’images : {len(items)}")

        if not items:
            print("⚠️ Aucune image à évaluer.")
            continue

        items.sort(key=lambda x: x[1], reverse=True)  # tri par homogénéité décroissante
        sorted_fnames = [fname for fname, _, _ in items]
        sorted_images_by_class[class_name] = sorted_fnames

        # Extraction des similarités triées
        similarities = [sim for _, sim, _ in items]

        # Calcul du point de rupture amélioré
        max_composite_gap = -1
        best_idx = -1

        for i in range(1, len(similarities) - 1):  # on évite les extrémités
            delta_before = abs(similarities[i] - similarities[i - 1])
            delta_after = abs(similarities[i] - similarities[i + 1])
            composite_gap = delta_before + delta_after

            if composite_gap > max_composite_gap:
                max_composite_gap = composite_gap
                best_idx = i

        if best_idx != -1:
            sim_before = similarities[best_idx - 1]
            sim_target = similarities[best_idx]
            sim_after = similarities[best_idx + 1]

            discriminability_threshold = (sim_before + sim_after) / 2

            print(f"\n📌 Point de discriminabilité trouvé à : {discriminability_threshold:.4f}")
            print(f"🧭 Point de rupture : image au centre = '{items[best_idx][0]}'")
            print(f"   ↳ Δ avant = {abs(sim_target - sim_before):.4f}, Δ après = {abs(sim_target - sim_after):.4f}, "
                f"Δ total = {max_composite_gap:.4f}")

            # Affichage des écarts combinés pour chaque image (sauf extrémités)
            delta_summary = []
            print("\n📉 Écarts combinés (avant + après) pour chaque image :")
            for i in range(len(similarities)):
                fname = items[i][0]
                if i == 0:
                    delta = abs(similarities[i] - similarities[i + 1])
                    delta_summary.append((fname, delta, None, delta))  # (nom, Δ après, Δ avant, Δ total)

                elif i == len(similarities) - 1:
                    delta = abs(similarities[i] - similarities[i - 1])
                    delta_summary.append((fname, None, delta, delta))
                else:
                    delta_before = abs(similarities[i] - similarities[i - 1])
                    delta_after = abs(similarities[i] - similarities[i + 1])
                    delta_total = delta_before + delta_after
                    delta_summary.append((fname, delta_after, delta_before, delta_total))
            # Tri décroissant selon Δ total
            delta_summary.sort(key=lambda x: x[3], reverse=True)

            # Affichage trié
            print("\n📉 Écarts combinés triés (Δ total décroissant) :")
            for fname, delta_after, delta_before, delta_total in delta_summary:
                if delta_before is None:
                    print(f" - {fname} (tête)       : Δ après = {delta_after:.4f} | Δ total = {delta_total:.4f}")
                elif delta_after is None:
                    print(f" - {fname} (queue)      : Δ avant = {delta_before:.4f} | Δ total = {delta_total:.4f}")
                else:
                    print(f" - {fname}              : Δ avant = {delta_before:.4f} | Δ après = {delta_after:.4f} | Δ total = {delta_total:.4f}")
        else:
            print("\n⚠️ Pas assez de données pour calculer un point de discriminabilité.")


            # Images proches du seuil (±0.05)
            #border_images = [(fname, sim, qual) for fname, sim, qual in items
                            #if abs(sim - discriminability_threshold) < 0.05]

            #print("\n⚠️ Images proches de la frontière discriminative :")
            #for fname, sim, qual in border_images:
                #print(f" - {fname} | Homogénéité : {sim:.4f} | Qualité : {qual:.2f}")
        #else:
            #print("\n📌 Pas assez d'images pour calculer un point de discriminabilité.")


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
        return sorted_images_by_class


def augmenter_dataset(base_path, image_type, augment_path, classes_folders, seed, img_height, img_width, images_par_homogeneite=None):
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

        print(f"\n🔎 Traitement de la classe : {class_name}")

        top_selected = []
        if images_par_homogeneite:
            top_imgs = images_par_homogeneite.get(class_name, [])
            cutoff = int(len(top_imgs) * 2 / 40)
            top_selected = top_imgs[:cutoff]

            print(f"🏆 Top 2/3 des images les plus homogènes ({len(top_selected)} images) :")
            for img in top_selected:
                print(f"   - {img}")

        for fname in os.listdir(class_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(class_path, fname)

                if images_par_homogeneite:
                    if fname not in top_selected:
                        print(f"⛔ Ignorée (hors top 2/3) : {fname}")
                        continue
                    print(f"✅ Gardée (top 2/3) : {fname}")

                class_images[class_name].append(full_path)

    # 3. Calculer combien d’images ajouter par classe
    # On garde la trace du nombre total d’images originales pour chaque classe
    true_class_counts = {class_name: len(images_par_homogeneite.get(class_name, [])) for class_name in classes_folders}
    max_count = max(true_class_counts.values())

    for label, image_paths in class_images.items():
        current_total = true_class_counts.get(label, len(image_paths))  # total images originales
        current_filtered = len(image_paths)  # uniquement les top 2/3 retenues

        if current_total == max_count:
            # Si c’est la classe la plus représentée, on augmente de 5%
            extra_needed = max(1, int(current_total * 0.05))
        else:
            # Sinon, on égalise jusqu’à la classe majoritaire
            extra_needed = max_count - current_total

        print(f"🔧 {label} → Ajout de {extra_needed} images (sur {current_filtered} images sélectionnées)")

        save_dir = os.path.join(augment_path, label)
        os.makedirs(save_dir, exist_ok=True)

        # 4. Générer des images augmentées
        for i in tqdm(range(extra_needed), desc=""):
            path = random.choice(image_paths)
            image = Image.open(path).convert("RGB").resize((img_height, img_width))
            aug_image = augmentation(image)

            save_path = os.path.join(save_dir, f"aug_{i}.jpg")
            aug_image.save(save_path)

            print(f"📈 Augmentation → Image source : {os.path.basename(path)} → Fichier généré : {os.path.basename(save_path)}")