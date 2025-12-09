import os
import random
import cv2
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from tqdm import tqdm

# -----------------------------
# PARAMÉTEREK
# -----------------------------
DATASET_DIR = "E:/MainCamera/dataset"
OUTPUT_DIR = "E:/MainCamera/classified_images" # Ide mentjük a tesztelt képeket
TRAINING_IMAGE_SAVE_DIR = "E:/MainCamera/training_source_images" # Ide mentjük a tanításhoz felhasznált forrásképeket

LR = 1e-3
EPOCHS = 20
import os
import random
import cv2
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from tqdm import tqdm

# -----------------------------
# PARAMÉTEREK
# -----------------------------
DATASET_DIR = "E:/MainCamera/dataset"
OUTPUT_DIR = "E:/MainCamera/classified_images" # Ide mentjük a tesztelt képeket
TRAINING_IMAGE_SAVE_DIR = "E:/MainCamera/training_source_images" # Ide mentjük a tanításhoz felhasznált forrásképeket

BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# -----------------------------
# 1. Videóból random képkiválogatás (in-memory)
#    (Módosítva: képkockák kinyerése másodpercenként)
#    (Módosítva: képkockák kinyerése 5 másodpercenként, 0-val kezdve)
# -----------------------------
def extract_frames_at_interval(video_path, interval_sec=5):
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
    for second in range(0, int(duration_sec) + 1, interval_sec):
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
    def __init__(self, root):

        self.data = []  # (image, label) párok
        
        self.classes = {
            "A": 0,
            "B": 1
        }
        
        for cls_name, label in self.classes.items():
            cls_dir = os.path.join(root, cls_name)

            # Könyvtár létrehozása a kinyert képkockák mentéséhez
            save_dir_for_class = os.path.join(TRAINING_IMAGE_SAVE_DIR, cls_name)
            os.makedirs(save_dir_for_class, exist_ok=True)
            
            # --- Videók feldolgozása ---
            videos = [v for v in os.listdir(cls_dir) if v.endswith((".mp4", ".avi", ".mov"))]
            for v in tqdm(videos, desc=f"Processing videos in '{cls_name}'"):

                path = os.path.join(cls_dir, v)

                frames = extract_frames_at_interval(path, interval_sec=5)
                for i, frame in enumerate(frames):
                    self.data.append((frame, label))

                    # Képkocka mentése a forráskönyvtárba
                    # A frame RGB, de a cv2.imwrite BGR-t vár
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_name_without_ext = os.path.splitext(v)[0]
                    save_path = os.path.join(save_dir_for_class, f"{video_name_without_ext}_frame_{i}.jpg")
                    cv2.imwrite(save_path, frame_bgr)
            
            # --- Képek feldolgozása ---
            image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
            images = [img for img in os.listdir(cls_dir) if img.lower().endswith(image_extensions)]
            for img_name in tqdm(images, desc=f"Processing images in '{cls_name}'"):
                path = os.path.join(cls_dir, img_name)
                try:
                    image = cv2.imread(path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.data.append((image, label))
                except Exception as e:
                    print(f"Hiba a kép beolvasásakor: {path} - {e}")

        # Transzformáció (erős downscale)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((270, 480)),     # FullHD → kb. 1/4
            # Még egy 4× maxpool a CNN elején
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# -----------------------------
# 3. Egyszerű CNN
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.MaxPool2d(4),  # brutális downscale (FullHD → nagyon kicsi)

            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# 4. Dataset betöltése
# -----------------------------
dataset = FootballDataset(DATASET_DIR)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)


# -----------------------------
# 5. Modell, loss, optimizer
# -----------------------------
model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


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

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")


# -----------------------------
# 7. Tesztelés
# -----------------------------
model.eval()

# Hozzuk létre a kimeneti mappákat, ha még nem léteznek
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# A címkéből a mappa nevére való leképezés
class_names = {v: k for k, v in dataset.classes.items()}
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
torch.save(model.state_dict(), "model_football.pth")
print("Model saved as model_football.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# -----------------------------
# 1. Videóból random képkiválogatás (in-memory)
#    (Módosítva: képkockák kinyerése másodpercenként)
#    (Módosítva: képkockák kinyerése 5 másodpercenként, 0-val kezdve)
# -----------------------------
def extract_frames_at_interval(video_path, interval_sec=5):
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
    for second in range(0, int(duration_sec) + 1, interval_sec):
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
    def __init__(self, root):

        self.data = []  # (image, label) párok
        
        self.classes = {
            "A": 0,
            "B": 1
        }
        
        for cls_name, label in self.classes.items():
            cls_dir = os.path.join(root, cls_name)
            
            # --- Videók feldolgozása ---
            videos = [v for v in os.listdir(cls_dir) if v.endswith((".mp4", ".avi", ".mov"))]
            for v in tqdm(videos, desc=f"Processing videos in '{cls_name}'"):

                path = os.path.join(cls_dir, v)

                frames = extract_frames_at_interval(path, interval_sec=5)
                for i, frame in enumerate(frames):
                    self.data.append((frame, label))

                    # Képkocka mentése a forráskönyvtárba
                    # A frame RGB, de a cv2.imwrite BGR-t vár
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_name_without_ext = os.path.splitext(v)[0]
                    save_path = os.path.join(save_dir_for_class, f"{video_name_without_ext}_frame_{i}.jpg")
                    cv2.imwrite(save_path, frame_bgr)
            
            # --- Képek feldolgozása ---
            image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
            images = [img for img in os.listdir(cls_dir) if img.lower().endswith(image_extensions)]
            for img_name in tqdm(images, desc=f"Processing images in '{cls_name}'"):
                path = os.path.join(cls_dir, img_name)
                try:
                    image = cv2.imread(path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.data.append((image, label))
                except Exception as e:
                    print(f"Hiba a kép beolvasásakor: {path} - {e}")

        # Transzformáció (erős downscale)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((270, 480)),     # FullHD → kb. 1/4
            # Még egy 4× maxpool a CNN elején
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# -----------------------------
# 3. Egyszerű CNN
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.MaxPool2d(4),  # brutális downscale (FullHD → nagyon kicsi)

            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# 4. Dataset betöltése
# -----------------------------
dataset = FootballDataset(DATASET_DIR)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)


# -----------------------------
# 5. Modell, loss, optimizer
# -----------------------------
model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


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

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")


# -----------------------------
# 7. Tesztelés
# -----------------------------
model.eval()

# Hozzuk létre a kimeneti mappákat, ha még nem léteznek
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# A címkéből a mappa nevére való leképezés
class_names = {v: k for k, v in dataset.classes.items()}
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
torch.save(model.state_dict(), "model_football.pth")
print("Model saved as model_football.pth")
