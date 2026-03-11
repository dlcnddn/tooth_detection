from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from inference import get_model
import os

app = FastAPI()

MODEL_ID = "cavity-pvbhl/1"
API_KEY = os.getenv("ROBOFLOW_API_KEY")

model = None

def get_loaded_model():
    global model

    if not API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY 환경변수가 설정되지 않았습니다.")

    if model is None:
        print("loading model...")
        model = get_model(
            model_id=MODEL_ID,
            api_key=API_KEY
        )
        print("model loaded")

    return model

@app.get("/")
def root():
    return {"status": "server running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    current_model = get_loaded_model()

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {
            "image_path": "",
            "image_width": 0,
            "image_height": 0,
            "detection_count": 0,
            "detections": [],
            "error": "이미지를 읽을 수 없습니다."
        }

    results = current_model.infer(image)
    result = results[0]

    response_data = {
        "image_path": file.filename,
        "image_width": result.image.width,
        "image_height": result.image.height,
        "detection_count": len(result.predictions),
        "detections": []
    }

    for pred in result.predictions:
        x = int(pred.x)
        y = int(pred.y)
        w = int(pred.width)
        h = int(pred.height)

        x1 = x - w // 2
        y1 = y - h // 2
        x2 = x + w // 2
        y2 = y + h // 2

        response_data["detections"].append({
            "label": pred.class_name,
            "confidence": round(float(pred.confidence), 4),
            "class_id": int(pred.class_id) if pred.class_id is not None else None,
            "center_x": x,
            "center_y": y,
            "width": w,
            "height": h,
            "box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            },
            "detection_id": pred.detection_id
        })

    return response_data
