from core.ocr_engine import LicensePlateRecognizer
import os

def main():
    # 1. Inicjalizacja (use_gpu=True jeśli masz NVIDIA CUDA, w przeciwnym razie False)
    recognizer = LicensePlateRecognizer(use_gpu=False)

    # 2. Ścieżka do przykładowego zdjęcia (zmień nazwę pliku na właściwą!)
    # Upewnij się, że masz jakiś plik w folderze 'data'
    test_image_path = "data/1.jpg"

    if not os.path.exists(test_image_path):
        print(f"Błąd: Nie znaleziono pliku {test_image_path}. Wrzuć zdjęcie do folderu data.")
        return

    # 3. Analiza
    print(f"Analizuję: {test_image_path}...")
    text, conf, timing = recognizer.analyze_image(test_image_path)

    # 4. Wynik
    print("-" * 30)
    print(f"Wynik OCR: {text}")
    print(f"Pewność:   {conf:.4f}")
    print(f"Czas:      {timing:.4f} s")
    print("-" * 30)

    # Sprawdzenie wymagań
    if timing > 0.6:
        print("UWAGA: Czas > 0.6s (limit dla 100 zdjęć w 60s może być zagrożony).")
    else:
        print("Czas OK.")

if __name__ == "__main__":
    main()