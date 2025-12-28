import os
import random
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

# -----------------------------
# PARAMÉTEREK
# -----------------------------
DATASET_DIR = "E:/MainCamera/dataset"
OUTPUT_DIR = "E:/MainCamera/classified_images" # Ide mentjük a tesztelt képeket
TRAINING_IMAGE_SAVE_DIR = "E:/MainCamera/training_source_images" # Ide mentjük a tanításhoz felhasznált forrásképeket

BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 7 # Megnöveljük az epochok számát

INTERVAL_SEC = 3  # Másodpercenként kinyert képkockák száma

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# -----------------------------
# 1. Videóból random képkiválogatás (in-memory)
#    (Módosítva: képkockák kinyerése másodpercenként)
#    (Módosítva: képkockák kinyerése 5 másodpercenként, 0-val kezdve)
# -----------------------------
def extract_frames_at_interval(video_path, interval_sec=INTERVAL_SEC):
    """Kinyeri a képkockákat a videóból a megadott időközönként (másodperc)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Hiba: A videó megnyitása sikertelen: {video_path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0 or fps == 0:
        cap.release()
        return []

    duration_sec = frame_count / fps

    frames = []
    # A videóból kiveszünk egy képkockát `interval_sec` másodpercenként, 0-tól kezdve.
    for second in range(0, int(duration_sec) + 1, INTERVAL_SEC):
        frame_index = int(second * fps)
        if frame_index < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
    cap.release()
    return frames


# -----------------------------
# 2. Egyedi dataset
# -----------------------------
class FootballDataset(Dataset):
    def __init__(self, files_to_process, class_map, training_image_save_root, transform=None):

        self.data = []  # (image, label) párok
        self.classes = class_map

        for file_path, cls_name, label in tqdm(files_to_process, desc=f"Processing files"):
            # Könyvtár létrehozása a kinyert képkockák mentéséhez
            save_dir_for_class = os.path.join(training_image_save_root, cls_name)
            os.makedirs(save_dir_for_class, exist_ok=True)

            if file_path.lower().endswith((".mp4", ".avi", ".mov")):
                frames = extract_frames_at_interval(file_path, interval_sec=INTERVAL_SEC)
                for i, frame in enumerate(frames):
                    self.data.append((frame, label))

                    # Képkocka mentése a forráskönyvtárba
                    # A frame RGB, de a cv2.imwrite BGR-t vár
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_name_without_ext = os.path.splitext(os.path.basename(file_path))[0]
                    save_path = os.path.join(save_dir_for_class, f"{video_name_without_ext}_frame_{i}.jpg")
                    cv2.imwrite(save_path, frame_bgr)
            elif file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                try:
                    image = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.data.append((image, label))
                except Exception as e:
                    print(f"Hiba a kép beolvasásakor: {file_path} - {e}")

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return self.transform(img), torch.tensor(label, dtype=torch.long)


# -----------------------------
# 3. Egyszerű CNN
# (Módosítva: Transzfer tanulás ResNet18-cal)
# -----------------------------
def create_resnet_model(num_classes=2):
    """
    Létrehoz egy előtanított ResNet18 modellt és lecseréli az utolsó rétegét.
    """
    # Előtanított ResNet18 modell betöltése
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Az utolsó, teljesen összekötött réteg (classifier) cseréje
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


# -----------------------------
# 4. Dataset betöltése
# -----------------------------

# --- Adatfájlok összegyűjtése és szétválasztása videók/képek alapján ---
all_files = []
class_map = {"A": 0, "B": 1}

for cls_name, label in class_map.items():
    cls_dir = os.path.join(DATASET_DIR, cls_name)
    for filename in os.listdir(cls_dir):
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png", ".bmp")):
            all_files.append((os.path.join(cls_dir, filename), cls_name, label))

random.shuffle(all_files)

split_idx = int(0.8 * len(all_files))
train_files = all_files[:split_idx]
test_files = all_files[split_idx:]

print(f"Összes fájl: {len(all_files)}")
print(f"Tanító fájlok száma: {len(train_files)}")
print(f"Tesztelő fájlok száma: {len(test_files)}")

# --- Transzformációk definiálása ---
# A tanító transzformáció adat-augmentációt is tartalmaz
train_transform = T.Compose([
    T.ToPILImage(), # A cv2 numpy array-t PIL Image-dzsé alakítjuk
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.Resize((256, 512), antialias=True), # Átméretezés 2 hatványára
    T.ToTensor(),
    # Nincs többé Grayscale, 3 csatornás képet használunk
    # Normalizálás az ImageNet-en tanított modellekhez
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# A teszt transzformáció nem változtat a képeken, csak előkészíti őket
test_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 512), antialias=True), # Átméretezés 2 hatványára
    T.ToTensor(),
    # Nincs többé Grayscale
    # Normalizálás
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Dataset példányok létrehozása ---
print("\n--- Tanító adathalmaz létrehozása ---")
train_ds = FootballDataset(train_files, class_map, TRAINING_IMAGE_SAVE_DIR, transform=train_transform)

print("\n--- Teszt adathalmaz létrehozása ---")
test_ds = FootballDataset(test_files, class_map, TRAINING_IMAGE_SAVE_DIR, transform=test_transform)

print(f"\nTanító képek száma: {len(train_ds)}")
print(f"Teszt képek száma: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)


# -----------------------------
# 5. Modell, loss, optimizer
# -----------------------------
model = create_resnet_model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# Tanulási ráta ütemező hozzáadása
scheduler = StepLR(optimizer, step_size=7, gamma=0.1) # 7 epoch után 10-edére csökkenti az LR-t


# -----------------------------
# 6. Training loop
# -----------------------------
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step() # Ütemező léptetése minden epoch végén

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")


# -----------------------------
# 7. Tesztelés
# -----------------------------
model.eval()

# Hozzuk létre a kimeneti mappákat, ha még nem léteznek
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# A címkéből a mappa nevére való leképezés
class_names = {v: k for k, v in class_map.items()}
for class_name in class_names.values():
    class_path = os.path.join(OUTPUT_DIR, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

correct = 0
total = 0
image_counter = 0

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testing and saving images"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Képek mentése a predikció alapján
        for i in range(imgs.size(0)):
            pred_class_name = class_names[preds[i].item()]
            true_class_name = class_names[labels[i].item()]
            
            # A fájlnév jelzi, hogy helyes volt-e a tipp, és mi volt az eredeti címke
            correctness = "correct" if pred_class_name == true_class_name else f"wrong_actual_{true_class_name}"
            save_path = os.path.join(OUTPUT_DIR, pred_class_name, f"img_{image_counter}_{correctness}.png")
            save_image(imgs[i], save_path)
            image_counter += 1

print("Accuracy:", correct / total)


# -----------------------------
# 8. Mentés
# -----------------------------
torch.save(model.state_dict(), "scene_selector_model.pth")
print("Model saved as scene_selector_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
