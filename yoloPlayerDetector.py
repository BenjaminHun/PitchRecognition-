import cv2
import torch
from ultralytics import YOLO

class YoloPlayerDetector:
    def __init__(self, input_video_path, start_time=0, end_time=None, model_name='yolov8s.pt'):
        self.input_video_path = input_video_path
        self.start_time = start_time
        self.end_time = end_time
        
        # CUDA (GPU) ellenőrzés és beállítás
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Használt eszköz: {self.device.upper()}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            
        # YOLO modell betöltése
        # Az első futtatáskor automatikusan letölti a 'yolov8s.pt' fájlt
        print(f"Modell betöltése ({model_name})...")
        self.model = YOLO(model_name)

    def process(self):
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            print("Hiba: Nem sikerült megnyitni a videót.")
            return

        # Videó paraméterek és időzítés
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(self.start_time * fps)
        end_frame = int(self.end_time * fps) if self.end_time else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print("Feldolgozás indítása...")
        
        while True:
            ret, frame = cap.read()
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            if not ret or (self.end_time and current_frame > end_frame):
                break

            # YOLO inferencia futtatása
            # classes=[0] -> Csak a 'person' osztályt detektáljuk (COCO datasetben a 0-ás ID)
            # device -> GPU használata a gyorsításhoz
            results = self.model(frame, device=self.device, classes=[0], verbose=False)

            # Eredmény vizualizációja (bounding boxok kirajzolása a képre)
            annotated_frame = results[0].plot()

            # Debug infó kiírása
            cv2.putText(annotated_frame, f"Device: {self.device.upper()} | Frame: {current_frame}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Player Detection', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()