import shutil
import os
import cv2
import numpy as np
import pika
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from core.ocr_engine import OCREngine
from core.database import save_result

app = FastAPI(title="Car Plate Recognition System")

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("Inicjalizacja OCREngine dla API...")
engine = OCREngine(yolo_path='best.pt')


@app.post("/analyze/sync")
async def analyze_sync(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Nieprawidłowy plik obrazu")

    text, conf, bbox = engine.process_frame(img)

    result = {
        "filename": file.filename,
        "plate": text if text else None,
        "confidence": round(conf, 2),
        "found": text is not None
    }

    if text:
        save_result(file.filename, text, conf)

    return result


@app.post("/analyze/async")
async def analyze_async(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='plate_tasks')

    message = json.dumps({"file_path": file_path, "filename": file.filename})

    channel.basic_publish(exchange='',
                          routing_key='plate_tasks',
                          body=message)

    connection.close()

    return {"status": "queued", "filename": file.filename, "message": "Zadanie wysłane do workera"}