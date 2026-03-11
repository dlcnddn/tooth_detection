from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import base64
import requests

app = FastAPI(title="Tooth Detection API")


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"message": "server running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능하다.")

    model_id = os.getenv("ROBOFLOW_MODEL_ID")
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not model_id or not api_key:
        raise HTTPException(status_code=500, detail="ROBOFLOW_MODEL_ID 또는 ROBOFLOW_API_KEY가 설정되지 않았다.")

    try:
        # model_id 예: tooth-detection/3
        image_bytes = await file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        url = f"https://detect.roboflow.com/{model_id}"
        params = {
            "api_key": api_key
        }

        response = requests.post(
            url,
            params=params,
            data=image_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=60,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Roboflow API 오류: {response.status_code} / {response.text}"
            )

        result = response.json()

        return {
            "filename": file.filename,
            "result": result
        }

    except HTTPException:
        raise
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="추론 요청 시간이 초과되었다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")
