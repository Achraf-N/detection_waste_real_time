# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies with retry logic and alternative mirrors
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

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install "uvicorn[standard]"
# Copy the current directory contents into the container at /app
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]