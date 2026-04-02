# 6-DoF Pose Estimation using OpenCV

This project demonstrates estimation of object pose (rotation and translation) from 2D image points using the Perspective-n-Point (PnP) algorithm.

## Overview

Given known 3D object points and their corresponding 2D image projections, the system estimates the camera-to-object transformation and visualizes the result by projecting 3D coordinate axes onto the image.

## Pipeline

- Define 3D object points
- Define corresponding 2D image points
- Construct camera intrinsic matrix
- Estimate pose using `solvePnP`
- Project 3D axes onto image using `projectPoints`

## Concepts Used

- Perspective projection
- Camera calibration model
- 3D–2D correspondence
- Rigid transformation (rotation + translation)

## Technologies

- Python
- OpenCV
- NumPy

## Output

- Rotation vector (rvec)
- Translation vector (tvec)
- Visualized 3D axes on image

## Run

```bash
pip install -r requirements.txt
python main.py
