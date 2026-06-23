# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy your application code and models into the container
COPY . .

# Hugging Face strictly requires applications to run on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]