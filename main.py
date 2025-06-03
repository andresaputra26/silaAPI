from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import tensorflow as tf
import json
import os
import uvicorn

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sesuaikan jika kamu punya domain spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model dan label
basedir = os.path.dirname(__file__)
model_path = os.path.join(basedir, "model", "gesture_mlp_model.h5")
label_path = os.path.join(basedir, "model", "label.json")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = tf.keras.models.load_model(model_path)

if not os.path.exists(label_path):
    raise FileNotFoundError(f"Label file not found: {label_path}")
with open(label_path, "r") as f:
    label_dict = json.load(f)
    label_list = [label for label, _ in sorted(label_dict.items(), key=lambda x: x[1])]

# Request schema
class LandmarkRequest(BaseModel):
    landmarks: list[float] = Field(..., description="Flat list of 42 float values")

# Prediction endpoint
@app.post("/predict")
async def predict(req: LandmarkRequest):
    try:
        if len(req.landmarks) != 42:
            raise HTTPException(status_code=400, detail="Expected 42 values for landmarks")

        input_data = np.array([req.landmarks], dtype=np.float32)
        prediction = model.predict(input_data)
        index = int(np.argmax(prediction))
        label = label_list[index]
        confidence = float(np.max(prediction))

        return {"label": label, "confidence": confidence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# 🚀 Run server directly if invoked (and also works on Railway)
port = int(os.environ.get("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)
