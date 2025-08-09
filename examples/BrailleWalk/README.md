# BrailleWalk USB Camera Prototype Server

This example provides a minimal backend service to support the BrailleWalk prototype using a USB camera (or any OpenCV-supported camera). It uses Ultralytics YOLO for detection, optional OCR for reading text, and offline TTS for accessibility. It exposes HTTP endpoints and an MJPEG stream that a React Native (Expo SDK 53) app can consume.

Key capabilities:
- USB camera input only (no smartphone camera required)
- Real-time YOLO object detection and scene interpretation
- Heuristic environment safety/mood analysis (safe / unsafe / emergency)
- Optional OCR (Tesseract) to read text and signs, converted to speech
- Offline TTS (pyttsx3) to speak detections
- MJPEG preview stream and REST API endpoints for use by the RN app
- Face-recognition placeholders for future 2FA (no hard dependency)

Note: This is an example/prototype. It does not modify core Ultralytics code and should not affect tests.

## Requirements

Python 3.9+ recommended.

Install dependencies:

```
pip install ultralytics opencv-python fastapi uvicorn[standard] pyttsx3
# Optional OCR
pip install pytesseract
# Optional: install Tesseract engine
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Ubuntu: sudo apt-get install tesseract-ocr

# Optional face recognition (not required; endpoints will be stubs if missing)
# pip install face-recognition
```

A YOLO model is expected at repo root `yolov8n.pt` (already present in this repository). You can change the model path in the server config.

## Run the server

From repository root:

```
python examples/BrailleWalk/server.py
```

This starts a FastAPI server at http://127.0.0.1:8000.

## Endpoints

- GET `/` – Basic status
- GET `/preview` – MJPEG stream of the current camera feed (view in browser or in-app WebView)
- GET `/detections` – JSON with latest detections and derived scene description
- GET `/safety` – JSON with computed safety/mood status
- POST `/speak` – Body: `{ "text": "..." }` to speak via TTS
- POST `/announce/toggle` – Enable/disable automatic voice announcements
- POST `/ocr` – Attempt OCR on the latest frame; returns extracted text and optionally speaks it
- POST `/face/enroll` – Placeholder for face enrollment (returns 501 if face-recognition not installed)
- POST `/face/verify` – Placeholder for face verification (returns 501 if face-recognition not installed)

## Safety/Mood Heuristic

Simple rule-based scoring from detections:
- Emergency signals if classes like `fire`, `knife`, `gun`, `stop_sign` detected
- Unsafe if high crowd density near, vehicles very close, or obstacles (e.g., `stairs`) directly ahead
- Safe otherwise

You can refine in `examples/BrailleWalk/server.py`.

## React Native (Expo) Integration Tips

- MJPEG preview: use a WebView to render `/preview`.
- Detections polling: fetch `/detections` every ~500ms–1s (or use WebSocket if you extend the server).
- TTS on device: You may use Expo Speech for on-device narration; or rely on the server’s TTS for blind-first mode.
- Voice commands: Start with a simple push-to-talk button using `react-native-voice`, then send text commands to `/speak` or custom endpoints.

Example fetch (TypeScript):

```ts
const baseUrl = 'http://127.0.0.1:8000';

async function getDetections() {
  const res = await fetch(`${baseUrl}/detections`);
  return res.json();
}

async function speak(text: string) {
  await fetch(`${baseUrl}/speak`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
}
```

To show the camera preview (MJPEG) in Expo:
- Easiest: `WebView` component pointing to `${baseUrl}/preview`.
- Alternatively, implement your own MJPEG renderer.

## Configuration

Edit values at the top of `server.py`:
- CAMERA_INDEX (default 0) – change to your USB camera index
- MODEL_PATH (default `yolov8n.pt`)
- INFERENCE_SIZE (e.g., 640)
- ANNOUNCE_INTERVAL_SEC (throttle voice announcements)

## Notes

- If OCR (Tesseract) is not installed, the OCR endpoint will return 501 (not implemented) rather than failing.
- Face recognition endpoints are placeholders for 2FA; you can wire a proper face pipeline later.
- For Raspberry Pi camera: swap out the VideoCapture initialization with the appropriate GStreamer or `cv2.CAP_V4L2` options.

## License
This example follows the Ultralytics repository license. Use responsibly.
