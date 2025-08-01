from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
MODEL_PATH = r"d:\model rob\model_resnet50 (3).h5"
MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["Healthy", "Late Blight"]
TARGET_SIZE = (224, 224)  # Resize image to match model input size

@app.get("/")
async def serve_homepage():
    return FileResponse("static/index.html")  # Serve index.html from the static folder

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize(TARGET_SIZE)
    image = np.array(image) / 255.0  # Normalize
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)*100
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
