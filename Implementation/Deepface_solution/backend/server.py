from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from deepface import DeepFace

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

history = []
HISTORY_SIZE = 7

class ImageData(BaseModel):
    image: str

def decode_image(base64_str):
    img_data = base64.b64decode(base64_str.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def normalize_emotions(emotions):
    mapping = {
        "fearful": "fear",
        "surprised": "surprise",
        "disgusted": "disgust"
    }

    result = {}
    for k, v in emotions.items():
        k = mapping.get(k, k)
        result[k] = float(v)

    return result

def smooth(emotions):
    history.append(emotions)
    if len(history) > HISTORY_SIZE:
        history.pop(0)

    avg = {}
    for h in history:
        for k, v in h.items():
            avg[k] = avg.get(k, 0) + v

    for k in avg:
        avg[k] /= len(history)

    return avg

@app.post("/analyze")
async def analyze(data: ImageData):
    try:
        img = decode_image(data.image)

        result = DeepFace.analyze(
            img,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='retinaface'
        )

        if isinstance(result, list):
            result = result[0]

        emotions = result["emotion"]

        # 🔥 ВАЖНО: нормализация
        emotions = normalize_emotions(emotions)

        # 🔥 сглаживание
        smoothed = smooth(emotions)

        # 🔥 определяем эмоцию
        top = max(smoothed, key=smoothed.get)
        confidence = smoothed[top]

        # 🔥 фильтры
        if confidence < 40:
            top = "neutral"

        if top == "angry" and confidence < 65:
            top = "neutral"

        return {
            "emotion": top,
            "confidence": float(confidence),
            "all": smoothed
        }

    except Exception as e:
        print("ERROR:", e)
        return {"emotion": "error", "confidence": 0, "all": {}}