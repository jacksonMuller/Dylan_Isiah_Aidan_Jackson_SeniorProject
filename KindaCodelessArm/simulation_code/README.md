# SO-101 Vision Pick-and-Place

This project uses MuJoCo, OpenCV, and inverse kinematics to detect an object and move the SO-101 arm to it.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to use real hardware mode, also install:

```bash
pip install lerobot
```

## Run

Simulation:

```bash
mjpython main.py
```

Real hardware:

```bash
mjpython main.py --mode real
```

MuJoCo scene only:

```bash
mjpython simulate.py
```

IK test:

```bash
mjpython test_ik.py
```

## Calibrate

Create the pixel-to-world homography:

```bash
mjpython calibration/calibrate.py
```

Real camera calibration:

```bash
mjpython calibration/calibrate.py --mode real
```

## Manual `g` Sequence

When running in simulation:

- Press `g` in the MuJoCo viewer
- The robot picks the object sideways
- It lifts it, moves it to the place location, sets it down, and opens the gripper

## Edit The Pick And Place Coordinates

In `config.yaml`:

```yaml
manual_grasp_target:
  x: 0.20
  y: 0.10
  pick_approach_z: 0.15
  pick_z: 0.02
  carry_z: 0.18
  place_x: 0.10
  place_y: -0.12
  place_approach_z: 0.18
  place_z: 0.02
  wrist_roll_deg: 90.0
  gripper_open_deg: 80.0
  gripper_closed_deg: 5.0
```

Meaning:

- `x`, `y`: starting pick location
- `place_x`, `place_y`: ending place location
- `pick_z`: pickup height
- `place_z`: place-down height
- `carry_z`: travel height
- `wrist_roll_deg`: sideways orientation
