import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import cv2
import os
import math

# -----------------------------
# KONFIGURÁCIÓ
# -----------------------------
VIDEO_PATH = 'E:/MainCamera/original_match_footage/raw_video.mkv' # A teljes meccsvideó elérési útja
MODEL_PATH = 'scene_selector_model.pth'  # A tanított modell elérési útja
OUTPUT_DIR = 'E:/MainCamera/classified_frames_by_model/'
CLASS_NAMES = ['A', 'B']  # Az osztályok nevei, a modell kimenetének megfelelően

# Feldolgozandó időintervallumok másodpercben (perc * 60 + másodperc)
FIRST_HALF_START_SEC = 31 * 60 + 9
FIRST_HALF_END_SEC = 78 * 60 + 8
SECOND_HALF_START_SEC = 94 * 60 + 41
SECOND_HALF_END_SEC = 144 * 60 + 7


def create_resnet_model(num_classes=2):
    """
    A tanító szkriptből átvett modell definíciója.
    Ennek pontosan meg kell egyeznie a betanított modell felépítésével.
    """
    # Előtanított ResNet18 modell betöltése (súlyok nélkül, csak az architektúra kell)
    # A load_state_dict fogja betölteni a mi finomhangolt súlyainkat.
    model = models.resnet18(weights=None)

    # Az utolsó, teljesen összekötött réteg (classifier) cseréje
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


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
    model = create_resnet_model(num_classes=len(CLASS_NAMES))
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Hiba: A modell fájl nem található itt: {MODEL_PATH}")
        print("Kérlek, ellenőrizd a MODEL_PATH változó értékét.")
        return

    model.to(device)
    model.eval()  # A modellt 'evaluation' módba állítjuk

    # --- Kép-transzformációk definiálása ---
    # FONTOS: Ezeknek meg kell egyezniük a tanító szkriptben használtakkal!
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 512), antialias=True),
        T.ToTensor(),
        # Normalizálás az ImageNet-en tanított modellekhez
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Videó feldolgozása ---
    print(f"\n--- Feldolgozás indul: {os.path.basename(VIDEO_PATH)} ---")

    try:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise IOError(f"A videó megnyitása sikertelen: {VIDEO_PATH}")
    except FileNotFoundError:
        print(f"Hiba: A videófájl nem található: {VIDEO_PATH}")
        return
    except IOError as e:
        print(e)
        return

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("Hiba: A videó FPS értéke nem olvasható.")
            return

        # Kimeneti alkönyvtár létrehozása a videó nevével
        video_output_dir = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(VIDEO_PATH))[0])
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"A kimeneti képek a '{video_output_dir}' mappába kerülnek.")

        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # A videó végére értünk

            timestamp_sec = frame_number / fps

            # Ellenőrizzük, hogy a képkocka a megadott időintervallumokba esik-e
            is_in_first_half = FIRST_HALF_START_SEC <= timestamp_sec <= FIRST_HALF_END_SEC
            is_in_second_half = SECOND_HALF_START_SEC <= timestamp_sec <= SECOND_HALF_END_SEC

            # Csak akkor dolgozzuk fel, ha a megfelelő időintervallumban van,
            # és kb. másodpercenként egy képkockát vizsgálunk.
            if (is_in_first_half or is_in_second_half) and frame_number % math.ceil(fps) == 0:
                
                # Kép előfeldolgozása és klasszifikálása
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = preprocess(frame_rgb)
                input_batch = input_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_batch)
                    
                    # Softmax-ot alkalmazunk a kimenetre, hogy valószínűségeket kapjunk
                    probabilities = F.softmax(output, dim=1)
                    
                    # A legnagyobb valószínűség (bizonyosság) és a hozzá tartozó index (predikció)
                    confidence, pred_idx = torch.max(probabilities, 1)

                predicted_class = CLASS_NAMES[pred_idx.item()]
                confidence_score = confidence.item()
                
                # Kép mentése a predikció és a bizonyosság alapján
                filename = f"{timestamp_sec:.2f}s_{predicted_class}_conf_{confidence_score:.2f}.jpg"
                output_path = os.path.join(video_output_dir, filename)
                cv2.imwrite(output_path, frame)
            frame_number += 1

    finally:
        # Biztosítjuk, hogy a videó erőforrás mindig felszabaduljon, még hiba esetén is.
        cap.release()

    # --- Takarítás ---
    cv2.destroyAllWindows()
    print("Feldolgozás befejezve.")

if __name__ == "__main__":
    main()