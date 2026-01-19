import os
import cv2
import xml.etree.ElementTree as ET
import time
from core.ocr_engine import OCREngine  # Importujemy naszą nową klasę

# --- KONFIGURACJA ---
IMAGES_DIR = 'Data'
ANNOTATIONS_FILE = 'annotations.xml'


def clean_expected_text(text):
    import re
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def main():
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(ANNOTATIONS_FILE):
        print("Brak folderu Data lub pliku XML.")
        return

    # 1. Inicjalizacja Silnika (ładuje modele raz)
    engine = OCREngine(yolo_path='best.pt')

    # 2. Przygotowanie listy plików z XML
    tree = ET.parse(ANNOTATIONS_FILE)
    root = tree.getroot()
    images = root.findall('image')
    # Sortowanie numeryczne
    images.sort(key=lambda x: int(''.join(filter(str.isdigit, x.get('name')))) if any(
        c.isdigit() for c in x.get('name')) else 0)

    total = 0
    correct = 0
    start_time_all = time.time()

    print("-" * 100)
    print(f"{'PLIK':<10} | {'WYNIK OCR':<15} | {'OCZEKIWANY':<15} | {'STATUS'} | {'CONF'}")
    print("-" * 100)

    for image_tag in images:
        filename = image_tag.get('name')
        file_path = os.path.join(IMAGES_DIR, filename)

        if not os.path.exists(file_path): continue

        # Pobieramy oczekiwany wynik z XML dla porównania
        box = image_tag.find('box')
        if box is None: continue
        attr = box.find("attribute[@name='plate number']")
        if attr is None or not attr.text: continue
        expected = clean_expected_text(attr.text)

        img = cv2.imread(file_path)
        if img is None: continue

        # --- UŻYCIE SILNIKA ---
        # To jest jedyna linijka, którą musisz wywołać w aplikacji głównej!
        detected_text, conf, bbox = engine.process_frame(img)

        # Logika wyświetlania
        if detected_text is None:
            detected_text = "---"
            status = "🚫 (YOLO)"
        else:
            # Porównanie (ignorując O/0)
            norm_res = detected_text.replace('0', '#').replace('O', '#')
            norm_exp = expected.replace('0', '#').replace('O', '#')

            if norm_res == norm_exp:
                status = "✅"
                correct += 1
            else:
                status = "❌"

        total += 1
        print(f"{filename:<10} | {detected_text:<15} | {expected:<15} | {status}   | {conf:.2f}")

        if total >= 100: break

    total_time = time.time() - start_time_all
    print("-" * 100)
    print(f"SKUTECZNOŚĆ KOŃCOWA: {(correct / total) * 100:.2f}%")
    print(f"ŚREDNI CZAS (całość): {total_time / total:.4f}s / zdjęcie")


if __name__ == "__main__":
    main()