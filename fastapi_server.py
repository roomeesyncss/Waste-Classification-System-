from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
from typing import Dict

# Initialize FastAPI app
app = FastAPI(
    title="Waste Classification API",
    description="API for classifying waste using MobileNetV2",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = 160
MODEL_PATH = 'models/MobileNetV2_best.h5'
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

model = None

def load_classification_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        model = load_model(MODEL_PATH)
        print("✅ MobileNetV2 model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def preprocess_image(image_data):

    try:
        # Convert to PIL Image
        if isinstance(image_data, bytes):
            image_pil = Image.open(io.BytesIO(image_data))
        else:
            image_pil = image_data

        # Convert to RGB if necessary
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        # Convert to numpy array
        img_array = np.array(image_pil)

        # Resize to model input size
        img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

        # Normalize pixel values
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)

        return img_batch
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")

def predict_waste_class(processed_image):
    """Make prediction on processed image."""
    try:
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)

        # Get predicted class
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]

        return predicted_class
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    success = load_classification_model()
    if not success:
        print("⚠️ Warning: Model failed to load. API will not work properly.")

@app.get("/")
async def root():
    return {
        "message": "Waste Classification API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Predict waste class from uploaded image."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image data
        image_data = await file.read()

        # Preprocess image
        processed_image = preprocess_image(image_data)

        # Make prediction
        predicted_class = predict_waste_class(processed_image)

        return {
            "prediction": predicted_class,
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_base64")
async def predict_base64_image(data: Dict[str, str]):
    """Predict waste class from base64 encoded image."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode base64 image
        base64_data = data.get("image", "")
        if not base64_data:
            raise HTTPException(status_code=400, detail="No image data provided")

        # Remove data URL prefix if present
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]

        # Decode base64
        image_bytes = base64.b64decode(base64_data)

        # Preprocess image
        processed_image = preprocess_image(image_bytes)

        # Make prediction
        predicted_class = predict_waste_class(processed_image)

        return {
            "prediction": predicted_class
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classes")
async def get_classes():
    """Get list of available classes."""
    return {
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)