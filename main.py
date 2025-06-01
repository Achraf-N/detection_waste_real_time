import torch
from PIL import Image
import io
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForImageClassification, AutoImageProcessor
import base64
import requests
import time
import logging
from datetime import datetime
import threading
import queue
import numpy as np
from typing import Dict, Optional
import asyncio 
app = FastAPI()

# Configuration
MODEL_PATH = "achraf123/waste_model"
FASTAPI_URL = "https://organic-detection-api-1.onrender.com/api/organic"

MIN_CONFIDENCE = 0.9
ORGANIC_PAUSE_TIME = 7.0
DEBOUNCE_TIME = 2.0

# Thread-safe last sent times
last_sent_times_lock = threading.Lock()
last_sent_times: Dict[str, float] = {"organic": 0, "non_organic": 0}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    logger.info("Model loaded successfully")
    
    # Warmup the model
    with torch.no_grad():
        dummy_input = processor(
            images=Image.new('RGB', (224, 224)), 
            return_tensors="pt"
        )
        if torch.cuda.is_available():
            dummy_input = {k: v.cuda() for k, v in dummy_input.items()}
        model(**dummy_input)
    logger.info("Model warmup completed")
    
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Global state
pause_until = 0
is_paused = False
last_result = None

# Thread-safe queues
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
api_queue = queue.Queue(maxsize=100)  # Limit to prevent memory issues

def classify_image(image: Image.Image) -> Dict[str, float]:
    """Classify an image and return results"""
    try:
        inputs = processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_prob, top_class = torch.max(probs, dim=1)
        
        return {
            "class": model.config.id2label[top_class.item()],
            "confidence": top_prob.item()
        }
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {"class": "error", "confidence": 0.0}

def inference_worker():
    """Worker thread for processing inference requests"""
    logger.info("Inference worker started")
    while True:
        frame = frame_queue.get()
        if frame is None:  # Exit signal
            logger.info("Inference worker shutting down")
            break
        
        try:
            result = classify_image(frame)
            result_queue.put(result)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            result_queue.put({"class": "error", "confidence": 0.0})

def api_worker():
    """Background worker for API calls"""
    logger.info("API worker started")
    while True:
        data = api_queue.get()
        if data == "STOP":
            logger.info("API worker shutting down")
            break
            
        try:
            current_time = time.time()
            class_name = data["class_name"]
            confidence = data["confidence"]
            
            if not isinstance(confidence, float) or not 0 <= confidence <= 1:
                logger.error(f"Invalid confidence value: {confidence}")
                continue
                
            if class_name == "organic":
                with last_sent_times_lock:
                    if current_time - last_sent_times["organic"] > DEBOUNCE_TIME:
                        try:
                            response = requests.post(
                                FASTAPI_URL,
                                params={"confidence": float(confidence)},
                                timeout=3.0
                            )
                            if response.status_code == 200:
                                last_sent_times["organic"] = current_time
                                logger.info(f"Sent {class_name} to API (confidence: {confidence:.2f})")
                            else:
                                logger.error(f"API returned status {response.status_code}")
                        except requests.RequestException as e:
                            logger.error(f"API request failed: {e}")
            else:
                logger.debug(f"Non-organic detected - not sending to API")
                
        except Exception as e:
            logger.error(f"API worker error: {e}")

# Start worker threads
infer_thread = threading.Thread(target=inference_worker, daemon=True)
api_thread = threading.Thread(target=api_worker, daemon=True)
infer_thread.start()
api_thread.start()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")

    global pause_until, is_paused

    try:
        while True:
            current_time = time.time()

            # PAUSE MODE - No processing at all
            if is_paused:
                if current_time >= pause_until:
                    is_paused = False
                    logger.info("Resuming normal operation")
                    await websocket.send_json({"status": "resumed"})
                else:
                    # Clear any pending work
                    while not frame_queue.empty():
                        try:
                            frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    while not result_queue.empty():
                        try:
                            result_queue.get_nowait()
                        except queue.Empty:
                            break

                    await websocket.send_json({
                        "status": "paused",
                        "remaining": max(0, pause_until - current_time)
                    })
                    await asyncio.sleep(0.1)  # Prevent busy waiting
                    continue

            # Receive base64 image from client
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                b64_str = data.split(",")[1] if "," in data else data
                img_data = base64.b64decode(b64_str)
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                image = image.resize((224, 224))
            except asyncio.TimeoutError:
                logger.warning("WebSocket receive timeout")
                continue
            except Exception as e:
                logger.error(f"Image receive error: {e}")
                await websocket.send_json({"error": "invalid_image"})
                continue

            # Push frame to inference queue if empty
            if frame_queue.empty():
                try:
                    frame_queue.put_nowait(image)
                except queue.Full:
                    logger.warning("Frame queue is full; dropping frame")
                    await websocket.send_json({"warning": "queue_full"})
                    continue

            # Get result from inference if available
            try:
                result = result_queue.get_nowait()
                class_name = "organic" if result["class"] == "LABEL_0" else "non_organic"
                confidence = result["confidence"]

                logger.info(f"Detected: {class_name} | Confidence: {confidence:.2f}")

                response = {
                    "label": class_name,
                    "confidence": confidence,
                    "timestamp": current_time
                }

                await websocket.send_json(response)

                # Handle organic detection
                if class_name == "organic" and confidence >= MIN_CONFIDENCE:
                    api_queue.put({
                        "class_name": "organic",
                        "confidence": confidence
                    })
                    pause_until = time.time() + ORGANIC_PAUSE_TIME
                    is_paused = True
                    logger.info(f"Pausing ALL predictions for {ORGANIC_PAUSE_TIME} seconds")
                    await websocket.send_json({
                        "status": "paused",
                        "resume_time": pause_until,
                        "remaining": ORGANIC_PAUSE_TIME
                    })
                    continue

                # Handle non-organic detection
                elif confidence >= MIN_CONFIDENCE:
                    api_queue.put({
                        "class_name": "non_organic",
                        "confidence": confidence
                    })

            except queue.Empty:
                # No result available yet
                pass

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("WebSocket connection closed")
        await websocket.close()

@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down workers")
    api_queue.put("STOP")
    frame_queue.put(None)  # Signal inference worker to stop
    
    # Wait for queues to clear
    time.sleep(0.5)
    
    # Force clear any remaining items
    while not api_queue.empty():
        try:
            api_queue.get_nowait()
        except queue.Empty:
            break
            
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_PATH,
        "is_paused": is_paused,
        "pause_until": pause_until,
        "pause_remaining": max(0, pause_until - time.time()) if is_paused else 0,
        "queue_sizes": {
            "frame_queue": frame_queue.qsize(),
            "result_queue": result_queue.qsize(),
            "api_queue": api_queue.qsize()
        }
    }
