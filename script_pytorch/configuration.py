import torch
import yaml 
import os
import script_pytorch.cnn as cnn

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def detect_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def update_device_in_config(filepath):
    # Charger ou créer la configuration
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # S'assurer que 'model_parameters' existe
    if 'model_parameters' not in config or not isinstance(config['model_parameters'], dict):
        config['model_parameters'] = {}

    #Recherche des valeurs des paramètres

    device = detect_device()
    pin_memory = device == "cuda"
    num_workers = calculate_num_workers()


    dataset_dir = config['dataset']['base_path']

    classes_exclues = config['dataset']['classes_exclues']

    # 🔹 Lister tous les dossiers du dataset sauf ceux à exclure
    classes_folders = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d not in classes_exclues
    ])

    num_classes = len(classes_folders)

    model = cnn.CNNIllustration(num_classes)
    #A remplacer par params de config (différencier height et width)
    #Egalement check 3
    image_shape=(3, 224, 224)

    if device == "cuda" :
        batch_size = trouver_batch_size_max_progressif(model, image_shape, num_classes, device)
    else :
        batch_size = 16

    config['model_parameters']['device'] = device
    config['model_parameters']['pin_memory'] = pin_memory
    config['model_parameters']['num_workers'] = num_workers
    config['model_parameters']['batch_size'] = batch_size

    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Fichier {filepath} mis à jour :")
    print(f"  device = {device}")
    print(f"  pin_memory = {pin_memory}")
    print(f"  num_workers = {num_workers}")
    print(f"  batch_size = {batch_size}")
    if device == "cuda":
        print("Nom du GPU :", torch.cuda.get_device_name(0))
    else :
        print("cpu")


def calculate_num_workers():
    cpu_count = os.cpu_count()
    return max(1, cpu_count // 2) if cpu_count else 1


#METHODE BATCH_SIZE

def trouver_batch_size_max(model, image_shape, num_classes, device="cuda", start=8, max_batch=1024):
    batch_size = start
    successful_batch = start

    model = model.to(device)
    model.eval()

    while batch_size <= max_batch:
        try:
            # Crée un batch factice d’images aléatoires
            inputs = torch.randn(batch_size, *image_shape).to(device)
            labels = torch.randint(0, num_classes, (batch_size,)).to(device)

            # Passage dans le modèle
            with torch.no_grad():
                outputs = model(inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)

            successful_batch = batch_size
            batch_size *= 2  # essaie un batch 2x plus gros
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                break
            else:
                raise e

    return successful_batch

def trouver_batch_size_max_progressif(
    model,
    image_shape,
    num_classes,
    device="cuda",
    start=8,
    step=8,
    max_test=1024
):
    model = model.to(device)
    model.train()  # train() si tu veux tester backward, sinon eval()

    batch_size = start
    best_batch = start

    def test_batch(batch_size):
        try:
            # Vider avant test
            torch.cuda.empty_cache()

            # Données artificielles
            inputs = torch.randn(batch_size, *image_shape, pin_memory=True).to(device, non_blocking=True)
            labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long, pin_memory=True).to(device, non_blocking=True)

            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)

            # 🔄 Si tu veux tester backward (entraînement réaliste) :
            loss.backward()
            # torch.optim.SGD(model.parameters(), lr=0.01).step()  # optionnel

            # Synchroniser pour capter erreurs CUDA au bon endroit
            torch.cuda.synchronize()

            # Nettoyage
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()
            return True

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ OOM pour batch={batch_size}")
                torch.cuda.empty_cache()
                return False
            else:
                raise e

    while batch_size <= max_test:
        if test_batch(batch_size):
            best_batch = batch_size
            batch_size += step
        else:
            break

    return best_batch


def trouver_batch_size_max_binaire(model, image_shape, num_classes, device="cuda", min_batch=1, max_batch=1024):
    model = model.to(device)
    model.eval()

    def test_batch(batch_size):
        try:
            inputs = torch.randn(batch_size, *image_shape).to(device)
            labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long).to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            return True
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                return False
            else:
                raise e

    best_batch = min_batch
    low = min_batch
    high = max_batch

    while low <= high:
        mid = (low + high) // 2
        if test_batch(mid):
            best_batch = mid
            low = mid + 1  # Cherche plus grand
        else:
            high = mid - 1  # Réduit la taille

    return best_batch





#TEST BATCH SIZE ISOLE

"""

num_classes = 10
model = cnn.CNNIllustration(num_classes)
image_shape=(3, 224, 224)
device="cuda"



batch_max = trouver_batch_size_max(
    model, 
    image_shape, 
    num_classes, 
    device)

print(f"✅ Batch size maximum supporté par le GPU : {batch_max}")

batch_max = trouver_batch_size_max_progressif(
    model,
    image_shape,
    num_classes,
    device,
    step=4
)
print(f"✅ Batch size maximal détecté (progressif) : {batch_max}")

batch_max = trouver_batch_size_max_binaire(
    model,
    image_shape,
    num_classes,
    device
)
print(f"✅ Batch size maximum stable : {batch_max}")

"""


