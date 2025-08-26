# Imports de la bibliothèque standard
import copy
import os
import shutil

# Imports de bibliothèques tierces
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from datetime import datetime

# Imports locaux
import script_pytorch.cnn as cnn

def preparer_dataset(classes_autorisees, root_dir, type_images, batch_size, seed, img_height, img_width, num_workers, pin_memory):

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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle = True, num_workers=num_workers, pin_memory= pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle = False, num_workers=num_workers, pin_memory= pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle = False, num_workers=num_workers, pin_memory= pin_memory)

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
            print("💾 Meilleur modèle sauvegardé.")
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

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 📌 Sauvegarde dans le dossier
    courbes_dir = "logs/entrainement/courbes"
    os.makedirs(courbes_dir, exist_ok=True)  # s'assure que le dossier existe
    fichier_courbe = os.path.join(courbes_dir, f"courbes_{timestamp}.png")

    plt.savefig(fichier_courbe)
    plt.close()

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