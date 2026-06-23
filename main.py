import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
# 👇 CHANGED: Swapped out full tensorflow for the standalone runtime
import tflite_runtime.interpreter as tflite

app = FastAPI(title="Waste Classification API")

# --- CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite frontend (Local testing)
        # "https://your-frontend-domain.vercel.app", <-- 👇 Add your production frontend URL here later!
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
def run_tflite_inference(interpreter, input_data: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]["index"])


# --- IMAGE PREPROCESSING ---
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.asarray(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


# --- API ENDPOINT ---
@app.post("/api/classify-waste")
async def classify_waste(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)

        # --- BASE MODEL ---
        base_prediction = run_tflite_inference(
            interpreter_base,
            processed_image
        )

        base_idx = int(np.argmax(base_prediction[0]))
        base_conf = float(base_prediction[0][base_idx])

        material = BASE_CLASSES[base_idx]

        response = {
            "status": "success",
            "primary_material": material,
            "primary_accuracy": f"{round(base_conf * 100, 2)}%"
        }

        # --- SUB MODELS ---
        if material == "Cardboard":
            prediction = run_tflite_inference(
                interpreter_cardboard,
                processed_image
            )

            sub_idx = int(np.argmax(prediction[0]))
            sub_conf = float(prediction[0][sub_idx])

            response.update({
                "sub_classification": CARDBOARD_CLASSES[sub_idx],
                "sub_accuracy": f"{round(sub_conf * 100, 2)}%"
            })

        elif material == "Plastic":
            prediction = run_tflite_inference(
                interpreter_plastic,
                processed_image
            )

            sub_idx = int(np.argmax(prediction[0]))
            sub_conf = float(prediction[0][sub_idx])

            response.update({
                "sub_classification": PLASTIC_CLASSES[sub_idx],
                "sub_accuracy": f"{round(sub_conf * 100, 2)}%"
            })

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )