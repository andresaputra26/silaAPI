from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import tensorflow as tf
import json
import os

app = FastAPI()

# Middleware untuk mengizinkan akses dari frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti sesuai domain frontend production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path ke model dan label
basedir = os.path.dirname(__file__)
model_path = os.path.join(basedir, "model", "gesture_mlp_model.h5")
label_path = os.path.join(basedir, "model", "label.json")

# Load model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = tf.keras.models.load_model(model_path)

# Load label
if not os.path.exists(label_path):
    raise FileNotFoundError(f"Label file not found: {label_path}")
with open(label_path, "r") as f:
    label_dict = json.load(f)
    label_list = [label for label, _ in sorted(label_dict.items(), key=lambda x: x[1])]

# Input schema
class LandmarkRequest(BaseModel):
    landmarks: list[float] = Field(..., description="Flat list of 42 float values")

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

        print(f"[INFO] Predicted: {label} with confidence {confidence:.4f}")
        return {"label": label, "confidence": confidence}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# ✅ Tambahkan bagian ini agar Railway tahu cara menjalankan
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # PORT akan disediakan oleh Railway
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
