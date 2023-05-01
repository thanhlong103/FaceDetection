# Face Detection and Tracking

This repository contains Python scripts for face detection and tracking using two different methods - `faceMesh` and `haarcascade_frontalface_default.xml`.

## Requirements

- Python 3.x
- OpenCV
- mediapipe
- imutils

## File Descriptions

- `facedetection.py`: This script uses `faceMesh` to detect faces in real-time from the webcam stream.
- `facetracking.py`: This script uses `haarcascade_frontalface_default.xml` to detect and track faces in real-time from the webcam stream.
- `faceMesh`: This folder contains the pre-trained `faceMesh` model used for face detection.
- `haarcascade_frontalface_default.xml`: This file contains the pre-trained Haar Cascade classifier used for face detection.

## How to Run

1. Clone the repository to your local machine.
2. Install the required libraries mentioned in the `Requirements` section.
3. Run `facedetection.py` or `facetracking.py` in your terminal or command prompt.

Example command:

```
python facedetection.py
```

## Acknowledgements

- The `faceMesh` model is part of the [Mediapipe](https://github.com/google/mediapipe) project by Google.
- The `haarcascade_frontalface_default.xml` file is part of the OpenCV library.
