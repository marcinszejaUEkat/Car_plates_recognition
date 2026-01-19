import xml.etree.ElementTree as ET
import os
import shutil
import random
import cv2

# --- KONFIGURACJA ---
XML_FILE = 'annotations.xml'
SOURCE_IMG_DIR = 'Data'  # Twój folder ze zdjęciami
OUTPUT_DIR = 'datasets'  # Tu powstanie struktura dla YOLO


def convert_bbox(size, box):
    """Zamienia (xmin, ymin, xmax, ymax) na (x_center, y_center, width, height) znormalizowane"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def main():
    # 1. Przygotuj foldery
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    # 2. Wczytaj XML
    tree = ET.parse(XML_FILE)
    root = tree.getroot()

    images_data = []  # Lista par (filename, bbox)

    # Parsowanie XML
    for image_tag in root.findall('image'):
        filename = image_tag.get('name')
        width = int(image_tag.get('width'))
        height = int(image_tag.get('height'))

        box_tag = image_tag.find('box')
        if box_tag is None: continue

        xtl = float(box_tag.get('xtl'))
        ytl = float(box_tag.get('ytl'))
        xbr = float(box_tag.get('xbr'))
        ybr = float(box_tag.get('ybr'))

        images_data.append({
            'filename': filename,
            'w': width,
            'h': height,
            'bbox': (xtl, xbr, ytl, ybr)  # xmin, xmax, ymin, ymax
        })

    # 3. Mieszanie i podział (80% train, 20% val)
    random.seed(42)
    random.shuffle(images_data)
    split_idx = int(len(images_data) * 0.8)
    train_set = images_data[:split_idx]
    val_set = images_data[split_idx:]

    print(f"Znaleziono {len(images_data)} obrazów.")
    print(f"Treningowe: {len(train_set)}, Walidacyjne: {len(val_set)}")

    # 4. Przetwarzanie
    def process_set(dataset, split_name):
        for item in dataset:
            src_path = os.path.join(SOURCE_IMG_DIR, item['filename'])
            if not os.path.exists(src_path):
                print(f"Brak pliku: {src_path}")
                continue

            # Kopiuj zdjęcie
            dst_img_path = os.path.join(OUTPUT_DIR, 'images', split_name, item['filename'])
            shutil.copy(src_path, dst_img_path)

            # Stwórz plik label txt
            # Nazwa pliku txt musi być taka sama jak jpg
            txt_filename = os.path.splitext(item['filename'])[0] + ".txt"
            dst_label_path = os.path.join(OUTPUT_DIR, 'labels', split_name, txt_filename)

            # Konwersja koordynatów
            # Box w XML jest: xtl(min), xbr(max), ytl(min), ybr(max)
            # Funkcja convert oczekuje: xmin, xmax, ymin, ymax
            # Ale moja funkcja convert_bbox oczekuje: xmin, xmax, ymin, ymax
            b = item['bbox']
            # box = (xmin, xmax, ymin, ymax)
            bb = convert_bbox((item['w'], item['h']), b)

            with open(dst_label_path, 'w') as f:
                # Class ID 0 (tablica), potem współrzędne
                f.write(f"0 {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

    process_set(train_set, 'train')
    process_set(val_set, 'val')

    print("Zakończono przygotowanie danych dla YOLO!")

    # 5. Tworzenie pliku konfiguracyjnego data.yaml
    yaml_content = f"""
path: {os.path.abspath(OUTPUT_DIR)} 
train: images/train
val: images/val

nc: 1
names: ['license_plate']
"""
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)
    print("Utworzono plik data.yaml")


if __name__ == "__main__":
    main()