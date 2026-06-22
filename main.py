import io
from pathlib import Path
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import tensorflow.lite as tflite

app = FastAPI(title="Waste Classification API")

# --- MODEL PATHS ---
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# --- LOAD MODELS ---
try:
    interpreter_base = tflite.Interpreter(
        model_path=str(MODELS_DIR / "model_base.tflite")
    )
    interpreter_cardboard = tflite.Interpreter(
        model_path=str(MODELS_DIR / "model_cardboard.tflite")
    )
    interpreter_plastic = tflite.Interpreter(
        model_path=str(MODELS_DIR / "model_plastic.tflite")
    )
    interpreter_base.allocate_tensors()
    interpreter_cardboard.allocate_tensors()
    interpreter_plastic.allocate_tensors()
    print("🚀 Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise e

# --- LABELS ---
BASE_CLASSES = ["Cardboard", "Plastic", "Glass", "Metal", "Paper"]
CARDBOARD_CLASSES = ["Grade A", "Grade B"]
PLASTIC_CLASSES = ["Grade A", "Grade B"]

# --- TFLITE INFERENCE ---
def run_tflite_inference(
    interpreter: tflite.Interpreter,
    input_data: np.ndarray
) -> np.ndarray:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])

# --- IMAGE PREPROCESSING ---
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.asarray(img).astype(np.float32)
    img_array /= 255.0
    return np.expand_dims(img_array, axis=0)

# --- API ENDPOINT ---
@app.post("/classify-waste")
async def classify_waste(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)

        base_prediction = run_tflite_inference(
            interpreter_base,
            processed_image
        )
        base_idx = np.argmax(base_prediction[0])
        base_accuracy = round(float(base_prediction[0][base_idx]) * 100, 2)  # ✅ Akurasi base model
        material = BASE_CLASSES[base_idx]

        if material == "Cardboard":
            prediction = run_tflite_inference(
                interpreter_cardboard,
                processed_image
            )
            sub_idx = np.argmax(prediction[0])
            sub_accuracy = round(float(prediction[0][sub_idx]) * 100, 2)  # ✅ Akurasi sub model
            return {
                "status": "success",
                "primary_material": material,
                "primary_accuracy": f"{base_accuracy}%",
                "sub_classification": CARDBOARD_CLASSES[sub_idx],
                "sub_accuracy": f"{sub_accuracy}%"
            }

        elif material == "Plastic":
            prediction = run_tflite_inference(
                interpreter_plastic,
                processed_image
            )
            sub_idx = np.argmax(prediction[0])
            sub_accuracy = round(float(prediction[0][sub_idx]) * 100, 2)  # ✅ Akurasi sub model
            return {
                "status": "success",
                "primary_material": material,
                "primary_accuracy": f"{base_accuracy}%",
                "sub_classification": PLASTIC_CLASSES[sub_idx],
                "sub_accuracy": f"{sub_accuracy}%"
            }

        return {
            "status": "success",
            "primary_material": material,
            "primary_accuracy": f"{base_accuracy}%"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )