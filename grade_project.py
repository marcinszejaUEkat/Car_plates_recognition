import os
import cv2
import xml.etree.ElementTree as ET
import time
import sys
from core.ocr_engine import OCREngine

# --- KONFIGURACJA ---
IMAGES_DIR = 'Data'
ANNOTATIONS_FILE = 'annotations.xml'
YOLO_PATH = 'best.pt'  # Upewnij się, że best.pt jest w głównym folderze


# --- FUNKCJA OCENIAJĄCA (Zgodna z wymaganiami) ---
def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    """
    Calculates the final grade based on license plate OCR accuracy and processing time.
    Parameters:
    - accuracy_percent: OCR accuracy as a percentage (0–100)
    - processing_time_sec: total time to process 100 images in seconds
    Returns:
    - Grade on a scale from 2.0 to 5.0 (rounded to the nearest 0.5)
    """

    # 1. Check minimum requirements
    # Jeśli dokładność < 60% LUB czas > 60s -> Ocena 2.0
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0

    # 2. Normalize accuracy: 60% -> 0.0, 100% -> 1.0
    accuracy_norm = (accuracy_percent - 60) / 40

    # 3. Normalize time: 60s -> 0.0, 10s -> 1.0
    # Uwaga: im szybciej (mniej sekund), tym lepiej.
    # Jeśli czas < 10s, norm może wyjść > 1.0, przycinamy logicznie do max 1.0
    if processing_time_sec < 10: processing_time_sec = 10
    time_norm = (60 - processing_time_sec) / 50

    # 4. Compute weighted score
    score = 0.7 * accuracy_norm + 0.3 * time_norm

    # 5. Calculate grade
    grade = 2.0 + 3.0 * score

    # 6. Round to the nearest 0.5
    return round(grade * 2) / 2


def clean_expected_text(text):
    import re
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def main():
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(ANNOTATIONS_FILE):
        print("Brak folderu Data lub pliku XML.")
        return

    # Inicjalizacja silnika (nie wliczamy tego do czasu przetwarzania zdjęć)
    print("Inicjalizacja silnika AI...")
    try:
        engine = OCREngine(yolo_path=YOLO_PATH)
    except Exception as e:
        print(f"Błąd inicjalizacji: {e}")
        print("Upewnij się, że plik 'best.pt' jest w folderze projektu!")
        return

    # Przygotowanie danych
    tree = ET.parse(ANNOTATIONS_FILE)
    root = tree.getroot()
    images = root.findall('image')
    # Sortowanie numeryczne
    images.sort(key=lambda x: int(''.join(filter(str.isdigit, x.get('name')))) if any(
        c.isdigit() for c in x.get('name')) else 0)

    # Bierzemy pierwsze 100 zdjęć (zgodnie z wymogiem funkcji oceniającej)
    # Jeśli masz mniej niż 100, ocena może być przekłamana, ale skrypt zadziała.
    test_set = images[:195]

    print(f"\nRozpoczynam test na {len(test_set)} zdjęciach...")
    print("-" * 60)

    correct = 0
    total = 0

    # --- START POMIARU CZASU (Timer "egzaminacyjny") ---
    start_test_time = time.time()

    for image_tag in test_set:
        filename = image_tag.get('name')
        file_path = os.path.join(IMAGES_DIR, filename)

        if not os.path.exists(file_path):
            total += 1  # Liczymy jako błąd
            continue

        # Pobranie Ground Truth
        box = image_tag.find('box')
        attr = box.find("attribute[@name='plate number']") if box else None
        if attr is None or not attr.text:
            total += 1
            continue
        expected = clean_expected_text(attr.text)

        img = cv2.imread(file_path)

        # --- GŁÓWNA ANALIZA ---
        detected_text, conf, bbox = engine.process_frame(img)
        # ----------------------

        # Weryfikacja
        status = "❌"
        if detected_text:
            norm_res = detected_text.replace('0', '#').replace('O', '#')
            norm_exp = expected.replace('0', '#').replace('O', '#')

            if norm_res == norm_exp:
                correct += 1
                status = "✅"

        total += 1
        # Opcjonalnie: printuj postęp co 10 zdjęć, żeby nie śmiecić
        if total % 10 == 0:
            print(f"Przetworzono {total}/{len(test_set)}...")

    # --- KONIEC POMIARU CZASU ---
    end_test_time = time.time()
    total_processing_time = end_test_time - start_test_time

    # OBLICZENIA KOŃCOWE
    accuracy_percent = (correct / total) * 100 if total > 0 else 0

    # Wyliczanie oceny
    final_grade = calculate_final_grade(accuracy_percent, total_processing_time)

    # RAPORT
    print("\n" + "=" * 40)
    print(f"       RAPORT KOŃCOWY")
    print("=" * 40)
    print(f"Liczba zdjęć:       {total}")
    print(f"Poprawne odczyty:   {correct}")
    print(f"Dokładność:         {accuracy_percent:.2f}%  (Wymagane: >60%)")
    print(f"Czas całkowity:     {total_processing_time:.2f}s (Wymagane: <60s)")
    print(f"Średni czas/foto:   {total_processing_time / total:.4f}s")
    print("-" * 40)

    # Kolorowy wynik w konsoli (jeśli terminal obsługuje ANSI)
    grade_color = "\033[92m" if final_grade >= 3.0 else "\033[91m"  # Zielony lub Czerwony
    reset_color = "\033[0m"

    print(f"OCENA KOŃCOWA:      {grade_color}{final_grade}{reset_color}")
    print("=" * 40)


if __name__ == "__main__":
    main()