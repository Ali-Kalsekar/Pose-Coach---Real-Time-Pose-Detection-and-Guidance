# Pose Coach
> Last automated login update: 2026-04-27 18:10:39

Pose Coach is a real-time pose replication and guidance tool built with OpenCV and MediaPipe. It compares a live webcam feed against a reference pose image, highlights body and hand joint differences, and shows coaching feedback to help the user match the target posture more accurately.

## Features

- Real-time webcam pose tracking
- Body and hand landmark detection using MediaPipe
- Reference pose extraction from an image
- Joint-angle comparison between reference and live poses
- On-screen coaching feedback such as increase, decrease, or hold
- Similarity scoring with smoothing for more stable feedback
- Visual legend and side-by-side reference preview

## How It Works

1. A reference image is loaded from `reference1.jpg`.
2. MediaPipe extracts body and hand landmarks from the reference pose.
3. The webcam stream is processed frame by frame.
4. Live joint angles are compared against the reference pose.
5. The app draws landmarks, highlights mismatched joints, and displays a similarity score.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- NumPy

Install dependencies with:

```bash
pip install opencv-python mediapipe numpy
```

## Run

Make sure your reference image exists in the project folder and update the path in `Pose_Detector.py` if needed.

```bash
python Pose_Detector.py
```

Press `Esc` to close the live window.

## Project Files

- `Pose_Detector.py` - Main pose coaching script
- `reference.jpg` - Sample reference image
- `reference1.jpg` - Default reference image used by the script

## Notes

- The current script uses `reference1.jpg` as the default reference pose.
- The webcam must be available for live tracking.
- For best results, use a clear reference pose with visible arms and hands.

## Possible Improvements

- Save feedback screenshots or session logs
- Add more pose templates
- Support multiple reference poses
- Add a calibration step for user height and camera distance
- Improve stability with temporal filtering and rep counting

## License

No license has been specified for this project.