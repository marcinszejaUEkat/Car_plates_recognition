import cv2
import numpy as np
import torch
import re
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class OCREngine:
    def __init__(self, yolo_path='best.pt', trocr_model_name='microsoft/trocr-base-printed'):
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Inicjalizacja OCREngine na: {self.device.upper()}")

        print(f"   -> Ładowanie YOLO: {yolo_path}")
        self.yolo = YOLO(yolo_path)

        print(f"   -> Ładowanie TrOCR: {trocr_model_name}")
        self.processor = TrOCRProcessor.from_pretrained(trocr_model_name)
        self.trocr = VisionEncoderDecoderModel.from_pretrained(trocr_model_name).to(self.device)

        print("✅ Silnik gotowy do pracy.")

    def preprocess_crop(self, crop):
        crop = cv2.copyMakeBorder(
            crop, 10, 10, 10, 10,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_crop)

    def run_ocr_on_crop(self, pil_image):
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.trocr.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def clean_and_fix(self, text):
        if not text: return ""

        text = text.upper().replace(' ', '').replace('-', '').replace('.', '').replace(':', '')

        if text.startswith("GST"): text = text[1:]
        if text.startswith("TNEL"): text = text[1:]
        if text.startswith("KKOS"): text = text[1:]
        if text.startswith("BC8"): text = text.replace("BC8", "CB4", 1)

        if text.startswith("PL"): text = text[2:]

        match = re.search(r'([A-Z]{2,3}[0-9]{2,5}[A-Z0-9]*)', text)
        if match:
            text = match.group(1)[:8]
        else:
            text = re.sub(r'[^A-Z0-9]', '', text)[:8]

        t = list(text)
        if len(t) < 4: return text

        to_digit = {'O': '0', 'I': '1', 'Z': '2', 'B': '8', 'S': '5', 'D': '0'}
        to_char = {'0': 'O', '1': 'I', '2': 'Z', '8': 'B', '5': 'S'}

        for i in range(min(2, len(t))):
            if t[i] in to_char: t[i] = to_char[t[i]]

        prefix_len = 2
        if len(t) > 2:
            c3 = t[2]
            is_letter = c3.isalpha() and c3 not in ['Z', 'I', 'O', 'B']
            is_silesia = (t[0] == 'S' and c3 in ['Z', 'R', 'C', 'M', 'T', 'B', 'K', 'L', 'I'])

            if is_letter or is_silesia:
                prefix_len = 3
                if t[2] in to_char: t[2] = to_char[t[2]]
            else:
                if t[2] in to_digit: t[2] = to_digit[t[2]]

        for i in range(prefix_len, len(t)):
            is_last = (i == len(t) - 1)

            if not is_last:
                if t[i] in to_digit: t[i] = to_digit[t[i]]
                if t[i] == 'Z': t[i] = '2'
                if t[i] == 'S': t[i] = '5'
            else:
                if t[i] == 'Q': t[i] = '0'
                if t[i] == '1' and t[i - 1] == '6': t[i] = 'T'

        return "".join(t)

    def process_frame(self, frame):
        results = self.yolo.predict(frame, conf=0.20, verbose=False)

        if len(results[0].boxes) == 0:
            return None, 0, None

        best_box = results[0].boxes[0]
        conf = float(best_box.conf[0])
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)

        plate_crop = frame[y1:y2, x1:x2]

        if plate_crop.size == 0:
            return None, 0, None

        pil_crop = self.preprocess_crop(plate_crop)
        raw_text = self.run_ocr_on_crop(pil_crop)
        final_text = self.clean_and_fix(raw_text)

        return final_text, conf, (x1, y1, x2, y2)