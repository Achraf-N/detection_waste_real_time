# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (OpenCV, image display libs)
RUN set -e; \
    for i in 1 2 3; do \
        apt-get update && \
        apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/* \
        && break; \
        sleep 5; \
    done || exit 1

# Copy only requirements first (leverages Docker layer caching)
COPY requirements.txt .

# Install Python packages and uvicorn
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install "uvicorn[standard]"

    
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='achraf123/waste_model', local_dir='waste_model', local_dir_use_symlinks=False)"

# Copy your app code
COPY . .

# Expose the port Render expects
EXPOSE 8000

# Set buffer for logs
ENV PYTHONUNBUFFERED=1

# Run your FastAPI app via uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
