import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import math

# --- Konfiguráció ---
VIDEO_DIR = 'E:/MainCamera/test_videos' # A könyvtár, amiben a videók vannak
MODEL_PATH = 'scene_selector_model.pth'  # A MainCamera/main.py által mentett modell
OUTPUT_DIR = 'E:MainCamera/classified_frames_by_model/'
CLASS_NAMES = ['A', 'B']  # Az osztályok nevei, a modell kimenetének megfelelően


class SimpleCNN(nn.Module):
    """
    A MainCamera/main.py fájlból átvett modell definíciója.
    Ennek pontosan meg kell egyeznie a betanított modell felépítésével.
    """
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.MaxPool2d(4),
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
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    """
    A fő feldolgozó függvény.
    """
    # --- Kimeneti mappa létrehozása ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"A kimeneti képek a '{OUTPUT_DIR}' mappába kerülnek.")

    # --- Eszköz (CPU/GPU) beállítása ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eszköz: {device}")

    # --- Modell betöltése ---
    print(f"Modell betöltése: {MODEL_PATH}")
    model = SimpleCNN(num_classes=len(CLASS_NAMES))
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Hiba: A modell fájl nem található itt: {MODEL_PATH}")
        print("Kérlek, ellenőrizd a MODEL_PATH változó értékét.")
        return

    model.to(device)
    model.eval()  # A modellt 'evaluation' módba állítjuk

    # --- Kép-transzformációk definiálása ---
    # FONTOS: Ezeknek meg kell egyezniük a MainCamera/main.py-ban használtakkal!
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((270, 480), antialias=True),
    ])

    # --- Videók keresése a könyvtárban ---
    video_extensions = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv')
    try:
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(video_extensions)]
    except FileNotFoundError:
        print(f"Hiba: A megadott könyvtár nem található: {VIDEO_DIR}")
        return

    if not video_files:
        print(f"Nem található videófájl a '{VIDEO_DIR}' könyvtárban.")
        return

    print(f"Talált videók: {len(video_files)} db")

    # --- Videók feldolgozása egyenként ---
    for video_filename in video_files:
        video_path = os.path.join(VIDEO_DIR, video_filename)
        print(f"\n--- Feldolgozás indul: {video_filename} ---")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Hiba: A videó megnyitása sikertelen: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("Hiba: A videó FPS értéke nem olvasható.")
            cap.release()
            continue

        # Kimeneti alkönyvtár létrehozása a videó nevével
        video_output_dir = os.path.join(OUTPUT_DIR, os.path.splitext(video_filename)[0])
        os.makedirs(video_output_dir, exist_ok=True)

        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # A videó végére értünk

            # Csak minden N-edik képkockát dolgozzuk fel, ami kb. 1 másodpercnek felel meg
            if frame_number % math.ceil(fps) == 0:
                timestamp_sec = frame_number / fps
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                input_tensor = preprocess(pil_image)
                input_batch = input_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_batch)

                _, pred_idx = torch.max(output, 1)
                predicted_class = CLASS_NAMES[pred_idx.item()]

                filename = f"{timestamp_sec:.2f}s_{predicted_class}.jpg"
                output_path = os.path.join(video_output_dir, filename)
                cv2.imwrite(output_path, frame)

            frame_number += 1

        cap.release()

    # --- Takarítás ---
    cv2.destroyAllWindows()
    print("Feldolgozás befejezve.")


if __name__ == "__main__":
    main()