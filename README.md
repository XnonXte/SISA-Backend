# SISA-Backend

FastAPI backend for hierarchical waste classification using TensorFlow Lite models.

## Requirements

- Python 3.10+

## Installation

Create and activate a virtual environment:

### Linux/macOS

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### Windows

```cmd
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
.
├── app.py
├── models/
│   ├── model_base.tflite
│   ├── model_cardboard.tflite
│   └── model_plastic.tflite
├── requirements.txt
└── README.md
```

## Run

```bash
uvicorn app:app --reload
```

API:

```
http://127.0.0.1:8000
```

Docs:

```
http://127.0.0.1:8000/docs
```

## Endpoint

### `POST /classify-waste`

Accepts an image file and returns the classification result.

Example:

```json
{
  "status": "success",
  "primary_material": "cardboard",
  "sub_classification": "grade A"
}
```

## Tech Stack

- FastAPI
- TensorFlow Lite Runtime
- NumPy
- Pillow
- Uvicorn