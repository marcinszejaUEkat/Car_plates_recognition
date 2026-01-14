import cv2
import easyocr
import time
import re
import numpy as np


class LicensePlateRecognizer:
    def __init__(self, use_gpu=False):
        print("Ładowanie modelu EasyOCR (High Quality Mode)...")
        # Na Macu M4 'gpu=False' oznacza użycie CPU, które jest potwornie szybkie.
        # 'gpu=True' wymagałoby specjalnej konfiguracji torch-mps,
        # ale CPU M4 spokojnie zrobi to w ułamku sekundy.
        self.reader = easyocr.Reader(['en'], gpu=False)
        print("Model załadowany.")

    def preprocess_image(self, image):
        if image is None:
            return None

        # --- KONFIGURACJA POD JAKOŚĆ (Dla M4) ---

        # 1. Resize - Ale teraz celujemy w wysoką jakość.
        # Szerokość 1600px pozwoli precyzyjnie odczytać litery.
        target_width = 1600
        h, w = image.shape[:2]

        if w > target_width:
            scale = target_width / w
            image = cv2.resize(image, (int(target_width), int(h * scale)), interpolation=cv2.INTER_AREA)

        # 2. Skala szarości
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 3. Obustronny filtr (Bilateral Filter)
        # Usuwa szum, ale zachowuje krawędzie liter (lepiej niż zwykły blur).
        # Jest wolniejszy, ale M4 to udźwignie.
        img_denoised = cv2.bilateralFilter(img_gray, 11, 17, 17)

        # 4. Adaptacyjny próg (Adaptive Threshold) lub CLAHE
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) świetnie wyciąga detale.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_contrasted = clahe.apply(img_denoised)

        return img_contrasted

    def clean_text(self, text):
        # Tylko duże litery i cyfry
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def analyze_image(self, image_path):
        start_time = time.time()

        img = cv2.imread(image_path)
        if img is None:
            return None, 0.0, 0.0

        processed_img = self.preprocess_image(img)

        # allowlist - blokujemy śmieci.
        results = self.reader.readtext(
            processed_img,
            detail=1,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        best_text = ""
        best_conf = 0.0

        for (bbox, text, conf) in results:
            cleaned = self.clean_text(text)

            # Polskie tablice: 7-8 znaków, ale szukamy szerzej 5-9
            if 5 <= len(cleaned) <= 9:
                # Logika punktacji: Długość ma znaczenie.
                # Preferujemy ciągi 7-8 znakowe nawet z nieco mniejszą pewnością
                # niż krótkie 5-znakowe "pewniaki" (często błędy).

                # Bonus punktowy za idealną długość tablicy (7 lub 8 znaków)
                score = conf
                if len(cleaned) == 7 or len(cleaned) == 8:
                    score += 0.2  # Promujemy poprawne długości

                if score > best_conf:
                    best_conf = score  # Zapisujemy score jako "conf" dla porównania
                    best_text = cleaned
                    # Przywracamy prawdziwą pewność do zmiennej conf (dla outputu)
                    best_conf = conf

        process_time = time.time() - start_time
        return best_text, best_conf, process_time