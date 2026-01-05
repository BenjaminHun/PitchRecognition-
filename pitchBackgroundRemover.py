from dataclasses import dataclass
import os
import cv2
import numpy as np



@dataclass
class BackgroundRemoverConfig:
    num_samples: int = 10
    epsilon_factor: float = 0.005
    min_pitch_area: int = 10000
    morphology_kernel_size: int = 3
    smooth_factor: float = 0.7


class PitchBackgroundRemover:
    def __init__(self, input_video_path, start_time, end_time, config: BackgroundRemoverConfig):
        self.input_video_path = input_video_path
        self.start_time = start_time
        self.end_time = end_time
        self.config = config

        self.cap = cv2.VideoCapture(self.input_video_path)
        if not self.cap.isOpened():
            raise IOError("Error: Could not open video.")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)       

        self.lower_green, self.upper_green = self.calculate_pitch_color_bounds()
        self.lower_green = self.lower_green.astype(np.uint8)
        self.upper_green = self.upper_green.astype(np.uint8)

    def calculate_pitch_color_bounds(self, num_samples=10):
        """Sample frames and estimate HSV color bounds for the pitch."""
        hsv_values = []
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, total_frames // num_samples)

        for i in range(0, total_frames, sample_interval):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if not ret:
                continue
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_values.append(hsv_frame.reshape(-1, 3))

        if not hsv_values:
            raise ValueError("No frames sampled for pitch color bounds.")

        hsv_values = np.concatenate(hsv_values, axis=0)
        lower_bound = np.percentile(hsv_values, 15, axis=0)
        upper_bound = np.percentile(hsv_values, 97, axis=0)
        return lower_bound, upper_bound

    def save_color_bounds(self, filepath="color_bounds.npz"):
        """Save the calculated color bounds for future use."""
        np.savez(filepath,
                lower_green=self.lower_green,
                upper_green=self.upper_green)

    def load_color_bounds(self, filepath="color_bounds.npz"):
        """Load previously calculated color bounds."""
        if os.path.exists(filepath):
            data = np.load(filepath)
            self.lower_green = data['lower_green']
            self.upper_green = data['upper_green']
            return True
        return False

    def enhance_detection(self, mask):
        """Enhance the detection mask using morphological operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def remove_lines_and_noise(self, mask, kernel_size=7, min_area=50, sep_kernel_size=3):
        """
        Eltávolítja a vonalakat és zajt gyors komponens-alapú szűréssel.
        A módszer:
        1. "Tisztított" maszk létrehozása (Separation): Kisebb nyitással szétválasztjuk a játékosokat a vonalaktól.
        2. "Mag" keresése (Core Detection): Nagyobb nyitással megtaláljuk a biztos játékosokat.
        3. Komponensek keresése a tisztított maszkon.
        4. Csak azokat a komponenseket tartjuk meg, amelyeknek van "magja".
        """
        # 1. "Tisztított" maszk létrehozása (Separation)
        # Ez szétválasztja a játékost a vonaltól, ha épp összeérnek, és eltünteti a vékony vonalakat.
        sep_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sep_kernel_size, sep_kernel_size))
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, sep_kernel)

        # 2. "Mag" keresése (Core Detection)
        # A biztos játékos-blokkok megtalálása.
        core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        core_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, core_kernel)

        # 3. Connected Components a TISZTÍTOTT maszkon
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)

        # 4. Azoknak a label-eknek a kiválasztása, amelyek átfedésben vannak a "maggal"
        valid_labels = np.unique(labels[core_mask == 255])
        valid_labels = valid_labels[valid_labels != 0]  # 0 a háttér

        # Opcionális: Terület alapú szűrés (zajszűrés)
        valid_labels = [l for l in valid_labels if stats[l, cv2.CC_STAT_AREA] >= min_area]

        # 5. Gyors maszk rekonstrukció Lookup Table (LUT) segítségével
        lut = np.zeros(num_labels, dtype=np.uint8)
        lut[valid_labels] = 255
        final_mask = lut[labels]

        # 6. Finomítás: Visszanövesztés az eredeti maszk határain belül
        # Mivel a cleaned_mask kicsit kisebb lehet, egy enyhe dilatációval korrigálunk.
        final_mask = cv2.dilate(final_mask, sep_kernel, iterations=1)
        final_mask = cv2.bitwise_and(final_mask, mask)

        return final_mask

    def process(self):
        """Remove background from video, keeping only the pitch."""
        start_frame = int(self.start_time * self.fps)
        end_frame = int(self.end_time * self.fps)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while True:
            ret, img = self.cap.read()
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(current_frame)
            if not ret or current_frame > end_frame:
                break

            # 1. Detect pitch area first
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            pitch_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
            
            blurred = cv2.GaussianBlur(pitch_mask, (121, 121), 0)
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 1. Pálya terület maszkja (ROI) - hogy a nézőteret kizárjuk
                pitch_area_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(pitch_area_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

                # 2. Előtér kinyerése: Ami NEM zöld (játékosok + vonalak) ÉS a pályán belül van
                non_green_mask = cv2.bitwise_not(pitch_mask)
                foreground_mask = cv2.bitwise_and(non_green_mask, pitch_area_mask)

                # 3. Vonalak eltávolítása morfológiai rekonstrukcióval
                # Paraméterek hangolása:
                # kernel_size: Növeld (pl. 9, 11), ha vastagabb vonalak maradnak. (Vigyázat: kis játékosok eltűnhetnek)
                # sep_kernel_size: Növeld (pl. 3, 5), ha a vonalak "hozzáragadnak" a játékosokhoz.
                # min_area: Növeld (pl. 100, 200), ha sok a kis zaj/pötty.
                final_mask = self.remove_lines_and_noise(foreground_mask, kernel_size=5, min_area=200, sep_kernel_size=5)

                # Eredmény megjelenítése
                result_img = cv2.bitwise_and(img, img, mask=final_mask)
                cv2.imshow('Players Only', result_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
