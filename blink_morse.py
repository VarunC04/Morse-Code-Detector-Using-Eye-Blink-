import argparse
import threading
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import cv2
import imutils
import keyboard
import numpy as np
from scipy.spatial import distance as dist
import mediapipe as mp

import constants
import morse_code


# This version replaces dlib with MediaPipe FaceMesh and uses cv2.VideoCapture.
# The detector only runs between /start and /stop calls from the backend.


@dataclass
class BlinkDetectorConfig:
    EYE_AR_THRESH: float = constants.EYE_AR_THRESH
    EYE_AR_CONSEC_FRAMES: int = constants.EYE_AR_CONSEC_FRAMES
    EYE_AR_CONSEC_FRAMES_CLOSED: int = constants.EYE_AR_CONSEC_FRAMES_CLOSED
    PAUSE_CONSEC_FRAMES: int = constants.PAUSE_CONSEC_FRAMES
    WORD_PAUSE_CONSEC_FRAMES: int = constants.WORD_PAUSE_CONSEC_FRAMES
    BREAK_LOOP_FRAMES: int = constants.BREAK_LOOP_FRAMES  # kept for compatibility, not used to stop

    def update(self, overrides: Dict[str, float]) -> None:
        for field in self.__dataclass_fields__:
            if field in overrides and overrides[field] is not None:
                setattr(self, field, overrides[field])


def eye_aspect_ratio(eye: np.ndarray) -> float:
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# MediaPipe FaceMesh eye landmark indices (left/right eyes)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


def landmarks_to_eye_array(landmarks, indices, image_shape):
    """Convert selected mediapipe landmarks to a (6, 2) numpy array in pixel coords."""
    h, w = image_shape[:2]
    eye = np.zeros((len(indices), 2), dtype="float")
    for i, idx in enumerate(indices):
        lm = landmarks[idx]
        eye[i] = [lm.x * w, lm.y * h]
    return eye


class BlinkMorseDetector:
    """Encapsulates the blink-to-Morse detection loop so we can run it as a service."""

    def __init__(
        self,
        shape_predictor_path: str,  # kept for compatibility, NOT used with mediapipe
        *,
        video_src: int = 0,
        display: bool = False,
        keyboard_output: bool = False,
    ):
        self.shape_predictor_path = shape_predictor_path
        self.video_src = video_src
        self.display = display
        self.keyboard_output = keyboard_output

        self.config = BlinkDetectorConfig()
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._vs: Optional[cv2.VideoCapture] = None
        self._face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None

        self._state_lock = threading.Lock()
        self._running = False
        self._current_morse = ""
        self._total_morse = ""
        self._translated_text = ""
        self._last_error: Optional[str] = None
        self._last_frame: Optional[np.ndarray] = None

    # ---------- Public API used by backend ----------

    def start(self, *, display: Optional[bool] = None) -> bool:
        if display is not None:
            self.display = display
        with self._state_lock:
            if self._running:
                return False
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._running = True
            self._last_error = None
        self._thread.start()
        return True

    def run(self, *, display: Optional[bool] = None) -> None:
        self.start(display=display)
        if self._thread:
            self._thread.join()

    def stop(self) -> bool:
        with self._state_lock:
            if not self._running or not self._stop_event:
                return False
            self._stop_event.set()
        if self._thread:
            self._thread.join()
        return True

    def is_running(self) -> bool:
        with self._state_lock:
            return self._running

    def update_config(self, overrides: Dict[str, float]) -> Dict[str, float]:
        with self._state_lock:
            self.config.update(overrides)
            current = self.config_dict
        return current

    @property
    def config_dict(self) -> Dict[str, float]:
        return asdict(self.config)

    def status(self) -> Dict[str, Optional[str]]:
        with self._state_lock:
            return {
                "running": self._running,
                "current_morse": self._current_morse,
                "total_morse": self._total_morse,
                "translated": self._translated_text,
                "config": self.config_dict,
                "error": self._last_error,
            }

    def get_last_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent frame for streaming to the frontend."""
        with self._state_lock:
            if self._last_frame is None:
                return None
            return self._last_frame.copy()

    # ---------- Internal loop ----------

    def _run(self) -> None:
        try:
            self._setup_detector_video()
            self._loop_camera()
        except Exception as exc:  # pylint: disable=broad-except
            with self._state_lock:
                self._last_error = str(exc)
        finally:
            self._cleanup()
            with self._state_lock:
                self._running = False

    def _setup_detector_video(self) -> None:
        # Initialize MediaPipe FaceMesh
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Initialize camera
        self._vs = cv2.VideoCapture(self.video_src, cv2.CAP_DSHOW)
        if not self._vs.isOpened():
            raise RuntimeError("Could not open video source")

    def _loop_camera(self) -> None:
        if not self._vs or not self._face_mesh:
            raise RuntimeError("Detector not initialized. Call start() first.")

        cfg = self.config

        counter = 0
        eyes_open_counter = 0
        closed_eyes = False
        word_pause = False
        paused = False

        total_morse = ""
        morse_word = ""
        morse_char = ""
        translation_dirty = False

        while not self._stop_event.is_set():
            ret, frame = self._vs.read()
            if not ret or frame is None:
                continue

            frame = imutils.resize(frame, width=450)

            # MediaPipe expects RGB input
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark

                    leftEye = landmarks_to_eye_array(landmarks, LEFT_EYE_IDX, frame.shape)
                    rightEye = landmarks_to_eye_array(landmarks, RIGHT_EYE_IDX, frame.shape)

                    left_eye_ar = eye_aspect_ratio(leftEye)
                    right_eye_ar = eye_aspect_ratio(rightEye)
                    eye_ar = (left_eye_ar + right_eye_ar) / 2.0

                    leftEyeHull = cv2.convexHull(leftEye.astype("int"))
                    rightEyeHull = cv2.convexHull(rightEye.astype("int"))
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    # ---- Blink / Morse logic ----
                    if eye_ar < cfg.EYE_AR_THRESH:
                        counter += 1
                        if counter >= cfg.EYE_AR_CONSEC_FRAMES:
                            closed_eyes = True
                        if not paused:
                            morse_char = ""
                    else:
                        eyes_open_counter += 1
                        # Long closed -> dash
                        if counter >= cfg.EYE_AR_CONSEC_FRAMES_CLOSED:
                            morse_word += "-"
                            total_morse += "-"
                            morse_char += "-"
                            counter = 0
                            closed_eyes = False
                            paused = True
                            eyes_open_counter = 0
                            translation_dirty = True
                        # Short closed -> dot
                        elif closed_eyes:
                            morse_word += "."
                            total_morse += "."
                            morse_char += "."
                            counter = 1
                            closed_eyes = False
                            paused = True
                            eyes_open_counter = 0
                            translation_dirty = True
                        # Pause between characters
                        elif paused and eyes_open_counter >= cfg.PAUSE_CONSEC_FRAMES:
                            morse_word += "/"
                            total_morse += "/"
                            morse_char = "/"
                            paused = False
                            word_pause = True
                            closed_eyes = False
                            eyes_open_counter = 0
                            if self.keyboard_output:
                                keyboard.write(morse_code.from_morse(morse_word))
                            morse_word = ""
                            translation_dirty = True
                        # Longer pause between words
                        elif word_pause and eyes_open_counter >= cfg.WORD_PAUSE_CONSEC_FRAMES:
                            total_morse += "¦/"
                            morse_char = ""
                            word_pause = False
                            closed_eyes = False
                            eyes_open_counter = 0
                            if self.keyboard_output:
                                keyboard.write(morse_code.from_morse("¦/"))
                            translation_dirty = True

                    cv2.putText(
                        frame,
                        f"EAR: {eye_ar:.2f}",
                        (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"{morse_char}",
                        (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        2,
                    )

            if self.display:
                cv2.imshow("Blink Morse Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # Only manual stop when running as a script with a window
            if key == ord("]"):
                self._stop_event.set()

            if translation_dirty:
                translated_text = morse_code.from_morse(total_morse)
                translation_dirty = False
            else:
                translated_text = self._translated_text

            self._update_frame_and_state(frame, morse_char, total_morse, translated_text)

        # Flush any remaining word on exit
        if self.keyboard_output and morse_word:
            keyboard.write(morse_code.from_morse(morse_word))
        self._update_frame_and_state(frame, morse_char, total_morse, morse_code.from_morse(total_morse))

    def _cleanup(self) -> None:
        if self.display:
            cv2.destroyAllWindows()
        if self._vs:
            self._vs.release()
            self._vs = None
        if self._face_mesh:
            self._face_mesh.close()
            self._face_mesh = None
        with self._state_lock:
            self._last_frame = None

    def _update_frame_and_state(
        self,
        frame: np.ndarray,
        current: str,
        total: str,
        translated: str,
    ) -> None:
        with self._state_lock:
            self._last_frame = frame.copy()
            self._current_morse = current
            self._total_morse = total
            self._translated_text = translated


def main() -> None:
    arg_par = argparse.ArgumentParser()
    # kept for compatibility with old CLI; ignored now
    arg_par.add_argument(
        "-p",
        "--shape-predictor",
        required=False,
        help="(ignored) path to facial landmark predictor (dlib version)",
    )
    arg_par.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening an OpenCV window",
    )
    args = vars(arg_par.parse_args())

    detector = BlinkMorseDetector(
        args.get("shape_predictor") or "",
        display=not args["no_display"],
        keyboard_output=True,
    )
    detector.run()
    results = detector.status()
    print("Morse Code:", results["total_morse"].replace("¦", " "))
    print("Translated:", results["translated"])


if __name__ == "__main__":
    main()
