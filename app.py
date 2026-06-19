import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import tflite_runtime.interpreter as tflite

app = FastAPI(title="Lightweight TFLite Waste Classification API")

# --- 1. INITIALIZE & ALLOCATE TFLITE INTERPRETERS ---
try:
    # Load .tflite files instead of .h5 files
    interpreter_base = tflite.Interpreter(model_path="models/model_base.tflite")
    interpreter_cardboard = tflite.Interpreter(model_path="models/model_cardboard.tflite")
    interpreter_plastic = tflite.Interpreter(model_path="models/model_plastic.tflite")

    # Allocate memory tensors for the models
    interpreter_base.allocate_tensors()
    interpreter_cardboard.allocate_tensors()
    interpreter_plastic.allocate_tensors()
    print("🚀 All TFLite models loaded and memory tensors allocated successfully!")
except Exception as e:
    print(f"❌ Error loading TFLite models: {e}")
    raise e

# --- 2. DEFINE LABELS ---
BASE_CLASSES = ["Cardboard", "Plastic", "Glass", "Metal", "Paper"]
CARDBOARD_CLASSES = ["Grade_A", "Grade_B"]
PLASTIC_CLASSES = ["Grade_A", "Grade_B"] 

# --- 3. HELPER FUNCTION FOR TFLITE INFERENCE ---
def run_tflite_inference(interpreter: tflite.Interpreter, input_data: np.ndarray) -> np.ndarray:
    """Injects data into a TFLite interpreter and returns prediction scores."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor value
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run prediction computation
    interpreter.invoke()
    
    # Extract prediction results
    return interpreter.get_tensor(output_details[0]['index'])

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocesses byte stream into matching TFLite shape (1, 224, 224, 3)"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    
    img_array = np.asarray(img).astype(np.float32)
    img_array = img_array / 255.0  # Common Teachable Machine normalization
    # If output accuracy is off, swap with line below:
    # img_array = (img_array / 127.5) - 1.0

    return np.expand_dims(img_array, axis=0)

# --- 4. API ENDPOINT ---
@app.post("/classify-waste")
async def classify_waste(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        
        # --- STEP 1: Base Model Inference ---
        base_prediction = run_tflite_inference(interpreter_base, processed_image)
        base_class_idx = np.argmax(base_prediction[0])
        predicted_base = BASE_CLASSES[base_class_idx]
        
        # --- STEP 2: Multi-stage Routing ---
        if predicted_base == "Cardboard":
            cb_prediction = run_tflite_inference(interpreter_cardboard, processed_image)
            cb_class_idx = np.argmax(cb_prediction[0])
            final_grade = CARDBOARD_CLASSES[cb_class_idx]
            
            return {
                "status": "success",
                "primary_material": "Cardboard",
                "sub_classification": final_grade
            }
            
        elif predicted_base == "Plastic":
            plastic_prediction = run_tflite_inference(interpreter_plastic, processed_image)
            plastic_class_idx = np.argmax(plastic_prediction[0])
            final_grade = PLASTIC_CLASSES[plastic_class_idx]
            
            return {
                "status": "success",
                "primary_material": "Plastic",
                "sub_classification": final_grade
            }
            
        else:
            # --- STEP 3: Handle structural errors for other waste types ---
            raise HTTPException(
                status_code=422, 
                detail={
                    "error": f"Unsupported waste type detected: {predicted_base}",
                    "detected_class": predicted_base
                }
            )
            
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)