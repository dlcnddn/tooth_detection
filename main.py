from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from functools import lru_cache
import tempfile
import os

app = FastAPI(title="Tooth Detection API")


@app.get("/")
def root():
    return {"message": "server running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@lru_cache(maxsize=1)
def get_model():
    """
    무거운 import와 모델 로드를 서버 시작 시점이 아니라
    첫 추론 요청 시점으로 미룬다.
    """
    try:
        from inference import get_model

        model_id = os.getenv("ROBOFLOW_MODEL_ID")
        api_key = os.getenv("ROBOFLOW_API_KEY")

        if not model_id or not api_key:
            raise ValueError("ROBOFLOW_MODEL_ID 또는 ROBOFLOW_API_KEY가 설정되지 않았다.")

        model = get_model(model_id=model_id, api_key=api_key)
        return model

    except Exception as e:
        raise RuntimeError(f"모델 로드 실패: {str(e)}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능하다.")

    suffix = os.path.splitext(file.filename)[1] if file.filename else ".jpg"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        model = get_model()
        result = model.infer(temp_path)

        return JSONResponse(
            content={
                "filename": file.filename,
                "result": result,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")

    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
