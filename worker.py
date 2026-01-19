import pika
import json
import cv2
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ocr_engine import OCREngine
from core.database import save_result


def main():
    print(" [Worker] Ładowanie modeli AI...")
    engine = OCREngine(yolo_path='best.pt')
    print(" [Worker] Modele gotowe. Oczekiwanie na zadania...")

    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='plate_tasks')

    def callback(ch, method, properties, body):
        try:
            data = json.loads(body)
            file_path = data['file_path']
            filename = data['filename']

            print(f" [x] Odebrano zadanie: {filename}")

            if not os.path.exists(file_path):
                print(f" [!] Błąd: Plik nie istnieje {file_path}")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            img = cv2.imread(file_path)
            text, conf, bbox = engine.process_frame(img)

            if text:
                print(f" [V] Wykryto: {text} ({conf:.2f})")
                save_result(filename, text, conf, status="SUCCESS")
            else:
                print(f" [X] Nie wykryto tablicy.")
                save_result(filename, "", 0.0, status="NO_PLATE")

        except Exception as e:
            print(f" [!] Błąd przetwarzania: {e}")

        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='plate_tasks', on_message_callback=callback)

    channel.start_consuming()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Przerwano działanie workera')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)