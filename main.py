"""
Gesture-Based Game Controller
==============================
Maps webcam-tracked hand gestures and head turns to virtual keyboard
presses, so you can drive a car (or move a character) using:

  - RIGHT hand OPEN   -> accelerate (holds 'w')
  - RIGHT hand CLOSED -> brake      (holds 's')
  - LEFT  hand OPEN   -> reverse    (holds 'r' by default - remap as needed)
  - HEAD turned LEFT  -> steer left  (holds 'a')
  - HEAD turned RIGHT -> steer right (holds 'd')

Design notes
------------
- Uses MediaPipe Hands + Face Mesh for landmark detection.
- Uses pydirectinput (not pyautogui) for key injection, because most
  games read DirectInput/raw scan codes and silently ignore the
  synthetic WM_KEYDOWN events pyautogui sends. pydirectinput uses
  ctypes SendInput with scan codes, which the vast majority of games
  (and Windows apps) do respond to.
- Calibration is mandatory before driving starts: neutral head pose,
  and hand-open/closed baselines are recorded per-session, since
  camera angle/lighting/hand size vary between people and setups.
- Every gesture uses hysteresis (different enter/exit thresholds) plus
  a short frame-count debounce, so a single noisy frame doesn't cause
  a key to stutter on/off. Keys are only pressed/released on an actual
  *state transition*, never spammed every frame.

Requirements
------------
    pip install opencv-python mediapipe pydirectinput numpy

Note: mediapipe currently supports Python 3.9-3.11 most reliably.
pydirectinput is Windows-only. On macOS/Linux, swap it for `pynput`
(see NOTE near KEY_BACKEND below) - key injection semantics differ
slightly and some games (esp. those with anti-cheat) may block
synthetic input entirely regardless of backend.

Run
---
    python gesture_controller.py

Controls while running:
    c  - (re)start calibration
    q  - quit
"""

import time
import sys
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

# ---------------------------------------------------------------------------
# KEY BACKEND
# ---------------------------------------------------------------------------
# pydirectinput = best compatibility with Windows games (DirectInput/SendInput)
# Swap this block for pynput on macOS/Linux.
try:
    import pydirectinput as kb
    kb.PAUSE = 0  # don't add pydirectinput's default delay after every call
    BACKEND = "pydirectinput"
except ImportError:
    from pynput.keyboard import Controller, Key
    _pynput_ctrl = Controller()

    class _PynputShim:
        """Minimal shim so the rest of the code can call kb.keyDown/keyUp
        the same way regardless of backend."""
        def keyDown(self, key):
            _pynput_ctrl.press(key)

        def keyUp(self, key):
            _pynput_ctrl.release(key)

    kb = _PynputShim()
    BACKEND = "pynput"

print(f"[gesture_controller] using key backend: {BACKEND}")

# ---------------------------------------------------------------------------
# CONFIG - remap keys here without touching logic below
# ---------------------------------------------------------------------------
KEY_ACCELERATE = "w"
KEY_BRAKE = "s"
KEY_REVERSE = "r"
KEY_LEFT = "a"
KEY_RIGHT = "d"

# Debounce: how many consecutive frames a state must hold before we
# trust it enough to press/release a key. Raise this if keys flicker.
DEBOUNCE_FRAMES = 3

# Head-turn hysteresis. ENTER threshold must be crossed to start a turn,
# EXIT threshold (smaller) must be crossed to release it. The gap
# between them prevents rapid on/off flicker right at the boundary.
YAW_ENTER = 0.06
YAW_EXIT = 0.03

# ---------------------------------------------------------------------------
# MEDIAPIPE SETUP
# ---------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# Face Mesh landmark indices used for yaw estimation
NOSE_TIP = 1
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]


def fingers_extended(landmarks, handedness_label):
    """Return count of extended fingers (0-5) for one detected hand.

    landmarks: list of 21 mediapipe NormalizedLandmark objects
    handedness_label: "Left" or "Right" as reported by MediaPipe
        (note: MediaPipe reports handedness from the CAMERA's
        perspective on an unmirrored image; we flip the frame for
        display, so this is already the person's actual hand)
    """
    count = 0
    # Thumb: moves sideways, not up/down, so compare x instead of y.
    # Direction flips depending on which hand it is.
    if handedness_label == "Right":
        if landmarks[FINGER_TIPS[0]].x < landmarks[FINGER_PIPS[0]].x:
            count += 1
    else:
        if landmarks[FINGER_TIPS[0]].x > landmarks[FINGER_PIPS[0]].x:
            count += 1

    # Other four fingers: extended if tip is above (smaller y than) the
    # PIP joint, in image-normalized coordinates (y grows downward).
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if landmarks[tip].y < landmarks[pip].y:
            count += 1

    return count


def compute_yaw_score(face_landmarks):
    """Head-turn indicator, robust to left/right position in frame
    (not just raw nose x). 0 == perfectly centered between eyes.
    Positive == nose shifted toward the person's right (their head
    turning toward the camera's left in a mirrored view), negative
    the opposite. Sign convention is resolved during calibration.
    """
    nose = face_landmarks[NOSE_TIP]
    left_eye = face_landmarks[LEFT_EYE_INNER]
    right_eye = face_landmarks[RIGHT_EYE_INNER]

    eye_mid_x = (left_eye.x + right_eye.x) / 2.0
    eye_dist = abs(right_eye.x - left_eye.x)
    if eye_dist < 1e-6:
        return 0.0
    return (nose.x - eye_mid_x) / eye_dist


class DebouncedSwitch:
    """A boolean switch that only flips after N consecutive frames agree,
    and only fires a callback on an actual transition (never re-fires
    the same state every frame)."""

    def __init__(self, on_enter, on_exit, debounce_frames=DEBOUNCE_FRAMES):
        self.state = False
        self.pending = False
        self.pending_count = 0
        self.debounce_frames = debounce_frames
        self.on_enter = on_enter
        self.on_exit = on_exit

    def update(self, raw_value: bool):
        if raw_value == self.pending:
            self.pending_count += 1
        else:
            self.pending = raw_value
            self.pending_count = 1

        if self.pending_count >= self.debounce_frames and self.pending != self.state:
            self.state = self.pending
            if self.state:
                self.on_enter()
            else:
                self.on_exit()


class ThreeWaySwitch:
    """For head yaw: three mutually exclusive states (left / center / right)
    with hysteresis so the boundary doesn't flicker."""

    def __init__(self, enter_thresh, exit_thresh):
        self.enter_thresh = enter_thresh
        self.exit_thresh = exit_thresh
        self.state = "center"

    def update(self, score):
        if self.state == "center":
            if score > self.enter_thresh:
                self.state = "right"
            elif score < -self.enter_thresh:
                self.state = "left"
        elif self.state == "right":
            if score < self.exit_thresh:
                self.state = "center"
        elif self.state == "left":
            if score > -self.exit_thresh:
                self.state = "center"
        return self.state


def calibrate(cap):
    """Interactive calibration: prompts the user through each pose and
    records baseline yaw center. Hand open/closed uses a fixed finger-
    count threshold (3+ extended = open, <=1 = closed), which is stable
    enough across hand sizes that per-user calibration isn't needed -
    but head yaw center absolutely varies with camera position, so we
    calibrate that live.
    """
    print("\n=== CALIBRATION ===")
    print("Look straight at the camera and stay still...")
    samples = []
    start = time.time()
    while time.time() - start < 3.0:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            samples.append(compute_yaw_score(lm))
        cv2.putText(frame, "Calibrating... look straight ahead", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Gesture Controller", frame)
        cv2.waitKey(1)

    yaw_center = float(np.mean(samples)) if samples else 0.0
    print(f"Calibration complete. Neutral yaw center = {yaw_center:.4f}")
    return yaw_center


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        sys.exit(1)

    yaw_center = calibrate(cap)

    # Switches - each owns the actual key press/release calls so state
    # transitions and key injection stay tightly coupled and can't drift
    # out of sync with each other.
    accel_switch = DebouncedSwitch(
        on_enter=lambda: kb.keyDown(KEY_ACCELERATE),
        on_exit=lambda: kb.keyUp(KEY_ACCELERATE),
    )
    brake_switch = DebouncedSwitch(
        on_enter=lambda: kb.keyDown(KEY_BRAKE),
        on_exit=lambda: kb.keyUp(KEY_BRAKE),
    )
    reverse_switch = DebouncedSwitch(
        on_enter=lambda: kb.keyDown(KEY_REVERSE),
        on_exit=lambda: kb.keyUp(KEY_REVERSE),
    )
    yaw_switch = ThreeWaySwitch(YAW_ENTER, YAW_EXIT)
    steer_state = {"left": False, "right": False}

    def set_steer(direction, active):
        if direction == "left" and active != steer_state["left"]:
            steer_state["left"] = active
            (kb.keyDown if active else kb.keyUp)(KEY_LEFT)
        if direction == "right" and active != steer_state["right"]:
            steer_state["right"] = active
            (kb.keyDown if active else kb.keyUp)(KEY_RIGHT)

    fps_times = deque(maxlen=30)

    print("\nRunning. Press 'c' to recalibrate, 'q' to quit.\n")

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)

            right_open = False
            right_closed = False
            left_open = False

            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand_lm, handedness in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness,
                ):
                    label = handedness.classification[0].label  # "Left"/"Right"
                    count = fingers_extended(hand_lm.landmark, label)
                    mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                    if label == "Right":
                        right_open = count >= 4
                        right_closed = count <= 1
                    else:
                        left_open = count >= 4

            accel_switch.update(right_open)
            brake_switch.update(right_closed and not right_open)
            reverse_switch.update(left_open)

            if face_results.multi_face_landmarks:
                lm = face_results.multi_face_landmarks[0].landmark
                score = compute_yaw_score(lm) - yaw_center
                direction = yaw_switch.update(score)
                set_steer("left", direction == "left")
                set_steer("right", direction == "right")

            # --- HUD ---
            fps_times.append(time.time() - t0)
            fps = 1.0 / (sum(fps_times) / len(fps_times) + 1e-6)
            status = (
                f"ACC:{accel_switch.state} BRK:{brake_switch.state} "
                f"REV:{reverse_switch.state} L:{steer_state['left']} "
                f"R:{steer_state['right']}  FPS:{fps:.0f}"
            )
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            cv2.imshow("Gesture Controller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                yaw_center = calibrate(cap)

    finally:
        # Always release any held keys on exit so the game doesn't get
        # stuck thinking a key is permanently down.
        for held_key in (KEY_ACCELERATE, KEY_BRAKE, KEY_REVERSE, KEY_LEFT, KEY_RIGHT):
            try:
                kb.keyUp(held_key)
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()