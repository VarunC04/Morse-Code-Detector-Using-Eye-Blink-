# Blink Morse

Blink Morse translates eye blinks captured from a webcam into Morse code and text. The original script now powers a small FastAPI backend plus a React dashboard so you can start/stop detection and watch the transcription in real time.

![Blinking Hello World](demo/sample.gif)

## Project Layout

```
blink_morse.py        # Reusable blink detector + CLI entry point
constants.py          # Detection thresholds
morse_code.py         # Morse ↔ text helpers
backend/main.py       # FastAPI service that wraps the detector
frontend/             # React (Vite) dashboard
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- [dlib shape predictor (68 landmarks)](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
  - Place the extracted `shape_predictor_68_face_landmarks.dat` file in the project root (next to `blink_morse.py`).  
    Alternatively set `SHAPE_PREDICTOR_PATH=/path/to/shape_predictor_68_face_landmarks.dat`.

## Python Environment & Backend

```bash
cd blink-morse
python -m venv .venv
.venv\Scripts\activate        # PowerShell / cmd on Windows
# or source .venv/bin/activate on macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

Run the API (hot reload for development):

```bash
uvicorn backend.main:app --reload
```

The server exposes:

- `GET /status` – returns `{running, current_morse, total_morse, translated, config, error}`
- `POST /start` – launches blink detection in a background thread (`started` indicates whether a new run began)
- `POST /stop` – stops detection and releases the camera (`stopped` indicates whether anything was running)
- `POST /config` – update any of the threshold values at runtime

Camera access and the dlib predictor are only initialised when `/start` is called and torn down on `/stop` or server shutdown.

## Frontend Dashboard

### Quick Start (Recommended)

To run both the frontend and backend together:

```bash
cd frontend
npm install
npm run dev:full
```

This will automatically start both the Python backend server and the React frontend. The backend will use the virtual environment if available, otherwise it will use the system Python.

### Running Separately

If you prefer to run them separately:

**Backend only:**
```bash
cd frontend
npm run dev:backend
```

**Frontend only:**
```bash
cd frontend
npm run dev
```

**Or manually:**
```bash
# Terminal 1: Start backend
cd blink-morse
uvicorn backend.main:app --reload

# Terminal 2: Start frontend
cd frontend
npm run dev
```

By default Vite serves on `http://localhost:5173` and expects the backend at `http://localhost:8000`.  
If you expose the API elsewhere set `VITE_API_BASE` before running the dev server, e.g.

```bash
VITE_API_BASE="http://localhost:9000" npm run dev
```

### Dashboard Features

- Start/Stop buttons that hit the backend endpoints
- Live status card for the current Morse character, accumulated Morse string, and translated text
- Config panel to tweak EAR / pause thresholds (saves via `POST /config`)
- Polling every 750 ms plus a lightweight log of the most recent translations
- Running/stopped indicator and error banner for backend issues

## Optional CLI Usage

You can still run the detector directly without the API:

```bash
python blink_morse.py -p shape_predictor_68_face_landmarks.dat
```

Add `--no-display` to keep OpenCV from opening a window.  
To quit, close your eyes for `constants.BREAK_LOOP_FRAMES` frames or press `]` while the OpenCV window is focused.

## Configuration

- Default thresholds live in `constants.py`.
- Runtime adjustments can be pushed through the dashboard or by calling `POST /config` manually:

```json
{
  "EYE_AR_THRESH": 0.26,
  "EYE_AR_CONSEC_FRAMES": 4,
  "EYE_AR_CONSEC_FRAMES_CLOSED": 12,
  "PAUSE_CONSEC_FRAMES": 25,
  "WORD_PAUSE_CONSEC_FRAMES": 35
}
```

## Built With

- dlib – face detector + landmark predictor
- OpenCV – webcam capture + drawing utilities
- imutils – VideoStream helper & convenience utilities
- FastAPI + Uvicorn – lightweight backend & background task management
- React + Vite – minimal frontend

## Inspiration

US Admiral Jeremiah Denton was taken prisoner during the Vietnam War and was forced to participate in a propaganda interview; he blinked his eyes in Morse code, spelling T-O-R-T-U-R-E to confirm that US POWs were being tortured. [[Wiki](https://en.wikipedia.org/wiki/Jeremiah_Denton#Vietnam_War)] [[Footage](https://youtu.be/rufnWLVQcKg)]

## Acknowledgments

Blink detection based off the tutorial from [PyImageSearch](https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib).
