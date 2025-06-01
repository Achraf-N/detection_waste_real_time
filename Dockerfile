# Use Python 3.9 with slim base image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files from current directory to /app in container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV MODEL_PATH="./final_model"  
ENV FASTAPI_URL="http://host.docker.internal:8000/api/organic"
ENV MIN_CONFIDENCE=0.9
ENV DEBOUNCE_TIME=2.0
ENV ORGANIC_PAUSE_TIME=3.0

# Make sure the detection script is executable
RUN chmod +x detection.py

# Run the application
CMD ["python", "detection.py"]
