from fastapi import FastAPI
from inference import IntentDetectionOnnxModel
import uvicorn


app = FastAPI(title="Intent Detection Service")

model = None


@app.on_event("startup")
def load_model():
    global model
    model = IntentDetectionOnnxModel()


@app.get("/predict")
async def predict(text: str):
    global model
    pred = model.pipeline(text)[0]
    return {"Intent": model.id2label[str(pred["label"])], "Score": float(pred["score"])}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

