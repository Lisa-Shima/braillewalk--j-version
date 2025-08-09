import os
import threading
import time
from typing import List, Optional, Tuple

import cv2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import face_recognition  # optional
except Exception:
    face_recognition = None

from ultralytics import YOLO

# -------------------------
# Configuration
# -------------------------
CAMERA_INDEX = int(os.getenv("BRAILLEWALK_CAMERA_INDEX", "0"))
MODEL_PATH = os.getenv("BRAILLEWALK_MODEL_PATH", os.path.join(os.getcwd(), "yolov8n.pt"))
INFERENCE_SIZE = int(os.getenv("BRAILLEWALK_INFERENCE_SIZE", "640"))
CONF_THRESH = float(os.getenv("BRAILLEWALK_CONF_THRESH", "0.25"))
ANNOUNCE_INTERVAL_SEC = float(os.getenv("BRAILLEWALK_ANNOUNCE_INTERVAL", "4.0"))
MJPEG_JPEG_QUALITY = int(os.getenv("BRAILLEWALK_JPEG_QUALITY", "70"))

# Safety heuristic keywords (by class names)
EMERGENCY_CLASSES = {"fire", "knife", "gun", "stop_sign"}
UNSAFE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "person", "stairs"}
CURRENCY_HINTS = ["USD", "RWF", "RWF", "FRW", "RWF", "Rwandan", "Franc", "FR", "$", "€", "£"]


# -------------------------
# Global State
# -------------------------
class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox_xyxy: List[int]
    distance_hint: Optional[str] = None


class DetectionsPayload(BaseModel):
    timestamp: float
    safety: str
    description: str
    detections: List[Detection]


class OCRPayload(BaseModel):
    text: str


class SpeakBody(BaseModel):
    text: str


app = FastAPI(title="BrailleWalk USB Camera Server", version="0.1.0")

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Unable to open camera index {CAMERA_INDEX}")

model = YOLO(MODEL_PATH)

_latest_frame_lock = threading.Lock()
_latest_frame = None  # type: Optional[Tuple[float, any]]  # (timestamp, frame BGR)

_latest_detections_lock = threading.Lock()
_latest_detections: Optional[DetectionsPayload] = None

_announce_enabled = True
_last_announce_time = 0.0

# TTS subsystem: background worker thread + queue to avoid blocking async endpoints
from queue import Empty, Queue

aSYNC_TTS_QUEUE: "Queue[str]" = Queue(maxsize=100)
_tts_thread: Optional[threading.Thread] = None


def _tts_worker():
    engine = None
    if pyttsx3 is not None:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
        except Exception:
            engine = None
    while True:
        try:
            text = aSYNC_TTS_QUEUE.get(timeout=0.5)
        except Empty:
            continue
        if text is None:  # poison pill for shutdown if ever used
            break
        if not text:
            continue
        if engine is None:
            print(f"[TTS Fallback] {text}")
            continue
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS Error] {e}: {text}")


def tts_enqueue(text: str):
    try:
        aSYNC_TTS_QUEUE.put_nowait(text)
    except Exception:
        # If queue is full, drop oldest silently to remain responsive
        try:
            _ = aSYNC_TTS_QUEUE.get_nowait()
        except Exception:
            pass
        try:
            aSYNC_TTS_QUEUE.put_nowait(text)
        except Exception:
            pass


def safety_and_description(classes: List[str]) -> Tuple[str, str]:
    # Simple heuristic scoring
    has_emergency = any(c in EMERGENCY_CLASSES for c in classes)
    if has_emergency:
        return "emergency", "Emergency hazard detected."

    # If a lot of crowded or vehicles -> unsafe
    crowd = classes.count("person")
    vehicles = sum(classes.count(c) for c in ["car", "truck", "bus", "motorcycle", "bicycle"])
    stairs = classes.count("stairs")

    if crowd >= 4 or vehicles >= 1 or stairs >= 1:
        desc_parts = []
        if crowd >= 4:
            desc_parts.append("crowded area")
        if vehicles >= 1:
            desc_parts.append("vehicles nearby")
        if stairs >= 1:
            desc_parts.append("stairs ahead")
        desc = ", ".join(desc_parts) if desc_parts else "potential obstacles ahead"
        return "unsafe", f"Caution: {desc}."

    return "safe", "Area appears safe."


def format_description(dets: List[Detection], safety: str) -> str:
    # Compose a voice-friendly description
    if not dets:
        return "No notable objects detected."
    top_n = sorted(dets, key=lambda d: d.confidence, reverse=True)[:5]
    items = [f"{d.class_name} {int(d.confidence * 100)} percent" for d in top_n]
    base = ", ".join(items)
    return f"{base}. Safety status: {safety}."


def detection_thread():
    global _latest_frame, _latest_detections, _last_announce_time
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        ts = time.time()
        with _latest_frame_lock:
            _latest_frame = (ts, frame.copy())

        # YOLO inference
        results = model.predict(source=frame, imgsz=INFERENCE_SIZE, conf=CONF_THRESH, verbose=False)
        det_list: List[Detection] = []
        classes_flat = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                names = r.names
                for b in r.boxes:
                    cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                    conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
                    xyxy = b.xyxy[0].tolist() if hasattr(b.xyxy, "tolist") else list(b.xyxy)
                    class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    classes_flat.append(class_name)
                    det_list.append(Detection(class_name=class_name, confidence=conf, bbox_xyxy=[int(x) for x in xyxy]))

        safety, _ = safety_and_description(classes_flat)
        description = format_description(det_list, safety)

        payload = DetectionsPayload(
            timestamp=ts,
            safety=safety,
            description=description,
            detections=det_list,
        )
        with _latest_detections_lock:
            _latest_detections = payload

        # Throttled announcements
        if _announce_enabled and (time.time() - _last_announce_time) >= ANNOUNCE_INTERVAL_SEC:
            if safety == "emergency":
                tts_enqueue("Emergency detected. Please be careful.")
            elif safety == "unsafe":
                tts_enqueue("Caution: potential hazards ahead.")
            else:
                # Lightly announce key object if available
                if det_list:
                    tts_enqueue(f"{det_list[0].class_name} ahead.")
            _last_announce_time = time.time()

        # Small sleep to balance CPU
        time.sleep(0.02)


def gen_mjpeg():
    try:
        while True:
            with _latest_frame_lock:
                frame_data = _latest_frame
            if frame_data is None:
                time.sleep(0.05)
                continue
            _, frame = frame_data
            # Encode as JPEG
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_JPEG_QUALITY]
            ok, jpg = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue
            jpg_bytes = jpg.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n")
    except GeneratorExit:
        # Client disconnected normally
        return
    except Exception:
        # Swallow disconnect-related errors to avoid noisy tracebacks on Windows
        return


@app.get("/")
def root():
    return {"app": "BrailleWalk USB Camera Server", "model": os.path.basename(MODEL_PATH), "camera": CAMERA_INDEX}


@app.get("/preview")
def preview():
    return StreamingResponse(gen_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/detections")
def get_detections():
    with _latest_detections_lock:
        payload = _latest_detections
    if payload is None:
        return JSONResponse({"status": "warming_up"})
    return JSONResponse(payload.model_dump())


@app.get("/safety")
def get_safety():
    with _latest_detections_lock:
        payload = _latest_detections
    if payload is None:
        return JSONResponse({"status": "warming_up"})
    return JSONResponse({"timestamp": payload.timestamp, "safety": payload.safety, "description": payload.description})


@app.post("/speak")
async def post_speak(body: SpeakBody):
    text = (body.text or "").strip()
    if not text:
        return JSONResponse({"ok": False, "error": "empty text"}, status_code=400)
    # Enqueue TTS to avoid blocking the event loop; return immediately
    tts_enqueue(text)
    return JSONResponse({"ok": True})


@app.post("/announce/toggle")
async def toggle_announce(request: Request):
    global _announce_enabled
    data = await request.json()
    enable = bool(data.get("enable", True))
    _announce_enabled = enable
    return JSONResponse({"ok": True, "announce_enabled": _announce_enabled})


@app.post("/ocr")
async def post_ocr():
    if pytesseract is None:
        return JSONResponse({"ok": False, "error": "pytesseract not installed"}, status_code=501)
    with _latest_frame_lock:
        frame_data = _latest_frame
    if frame_data is None:
        return JSONResponse({"ok": False, "error": "no frame"}, status_code=503)
    _, frame = frame_data
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    text = text.strip()
    if text:
        # currency hint quick pass
        lowered = text.lower()
        currency = any(h.lower() in lowered for h in CURRENCY_HINTS)
        if currency:
            tts_speak(f"Currency detected: {text}")
        else:
            tts_speak(text)
    return JSONResponse({"ok": True, "text": text})


@app.post("/face/enroll")
async def face_enroll():
    if face_recognition is None:
        return JSONResponse({"ok": False, "error": "face-recognition not installed"}, status_code=501)
    # Placeholder: Implement enrollment storage.
    return JSONResponse({"ok": True, "note": "Enrollment stub."})


@app.post("/face/verify")
async def face_verify():
    if face_recognition is None:
        return JSONResponse({"ok": False, "error": "face-recognition not installed"}, status_code=501)
    # Placeholder: Implement verification logic.
    return JSONResponse({"ok": True, "note": "Verification stub."})


def _start_threads():
    th = threading.Thread(target=detection_thread, daemon=True)
    th.start()


if __name__ == "__main__":
    import uvicorn

    _start_threads()
    # Warm-up TTS
    try:
        tts_speak("BrailleWalk server started.")
    except Exception:
        pass
    uvicorn.run(app, host="127.0.0.1", port=8000)
