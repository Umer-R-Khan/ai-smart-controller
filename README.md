# AI Smart Controller

Gesture-based game controller that maps webcam-tracked hand gestures and head turns to keyboard input.

## Features

- Right hand open: accelerate (`W`)
- Right hand closed: brake (`S`)
- Left hand open: reverse (`R`)
- Head turned left/right: steer (`A`/`D`)
- Calibration, hysteresis, and frame debouncing for steadier controls

## Setup

This project is intended for Windows and Python 3.9-3.11.

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Look straight at the camera during calibration. Press `c` to recalibrate and `q` to quit.

Synthetic keyboard input may be blocked by some games or anti-cheat systems. Use this project only with software where you are allowed to send input.
