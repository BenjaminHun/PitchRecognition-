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

    def remove_lines_and_noise(self, mask):
        """
        Eltávolítja a vonalakat és zajt egy robusztus morfológiai rekonstrukcióval.
        Ez a módszer a "magokból" építi vissza a játékosokat, a vonalakat pedig elhagyja.
        """
        # 1. "Mag" kép (marker) létrehozása:
        # Egy erős nyitás (erózió -> dilatáció) eltünteti a vékony vonalakat,
        # és csak a játékosok "vastag" magját hagyja meg.
        # A kernel mérete kritikus: elég nagynak kell lennie, hogy a vonalakat eltüntesse,
        # de elég kicsinek, hogy a távoli/kisebb játékosok magja megmaradjon.
        anchor_kernel_size = 15
        anchor_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (anchor_kernel_size, anchor_kernel_size))
        marker = cv2.morphologyEx(mask, cv2.MORPH_OPEN, anchor_kernel)

        # 2. Rekonstrukció:
        # A "mag"-ból (marker) kiindulva iteratívan "visszanövesztjük" az alakzatokat,
        # de csak az eredeti maszk (mask) határain belül.
        # A vonalak, mivel nincs magjuk, nem fognak rekonstruálódni.
        recon_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        reconstructed = marker
        while True:
            dilated = cv2.dilate(reconstructed, recon_kernel)
            reconstructed_new = cv2.bitwise_and(dilated, mask)
            # Ha nincs változás, a rekonstrukció kész.
            if np.array_equal(reconstructed, reconstructed_new):
                break
            reconstructed = reconstructed_new
            
        # Opcionális utótisztítás a kisebb zajokra, ha a rekonstrukció után maradnának.
        final_mask = cv2.morphologyEx(reconstructed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                
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
                
                contour_img = img.copy()
                cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 3)
                cv2.imshow('Largest Contour', contour_img)

                # Create mask from largest_contour
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                masked_img = cv2.bitwise_and(img, img, mask=mask)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cropped_img = masked_img[y:y+h, x:x+w]
                cv2.imshow('Cropped Result', cropped_img)

            #cv2.imshow('2. Blurred', blurred)
            #cv2.imshow('3. Thresholded', thresh)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
