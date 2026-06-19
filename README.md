# Waste Classification API

A FastAPI backend that uses a hierarchical TensorFlow Lite (TFLite) pipeline to classify waste materials.

## Features

- Built with FastAPI and TensorFlow Lite
- Lightweight inference using `.tflite` models
- Supports image upload via API
- Hierarchical multi-stage classification
- Interactive API documentation with Swagger UI

---

## Requirements

- Python 3.10 or newer
- pip
- Virtual environment (recommended)

---

## Installation

### 1. Create a Virtual Environment

#### Linux/macOS

```bash
python3.10 -m venv venv
source venv/bin/activate
```

#### Windows

```cmd
python -m venv venv
venv\Scripts\activate
```

---

### 2. Install Dependencies

```bash
pip install fastapi uvicorn numpy pillow tflite-runtime
```

Alternatively, install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Model Files

Place the following model files in the project root:

```text
model_base.tflite
model_cardboard.tflite
model_plastic.tflite
```

> **Note:** Ensure that the class order in each exported Teachable Machine model matches the corresponding class arrays defined in `app.py`.

---

## Running the Server

### Option 1 (Recommended)

```bash
uvicorn app:app --reload
```

### Option 2

```bash
python app.py
```

The server will start locally at:

```text
http://127.0.0.1:8000
```

---

## API Documentation

FastAPI automatically provides interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

You can use this interface to upload images and test the API directly from your browser.

---

## Endpoint

### `POST /classify-waste`

Accepts an image file through `multipart/form-data`.

#### Request Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `file` | Image file | Waste image to classify |

---

## Example Responses

### Successful Classification

#### Cardboard

```json
{
  "status": "success",
  "primary_material": "cardboard",
  "sub_classification": "grade A"
}
```

#### Plastic

```json
{
  "status": "success",
  "primary_material": "plastic",
  "sub_classification": "plastic type 1"
}
```

---

### Unsupported Waste Type

If the base model detects a material that is not currently supported (such as glass, metal, or paper), the API returns:

```json
{
  "detail": {
    "error": "Unsupported waste type detected: glass",
    "detected_class": "glass"
  }
}
```

---

### Invalid File Type

```json
{
  "detail": "Uploaded file must be an image."
}
```

---

## Project Structure

```text
.
├── app.py
├── model_base.tflite
├── model_cardboard.tflite
├── model_plastic.tflite
├── requirements.txt
└── README.md
```

---

## Tech Stack

- FastAPI
- TensorFlow Lite Runtime
- NumPy
- Pillow
- Uvicorn

---

## Notes

- Images are resized to **224 × 224** before inference.
- Pixel values are normalized to the range **[0, 1]**.
- Classification is performed in two stages:
  1. Base model predicts the material type.
  2. Material-specific models provide finer classification for supported classes.
- Currently, detailed classification is available only for:
  - Cardboard
  - Plastic