# Legacy scripts

These scripts were written before the code was reorganised into `code/core` and `code/detectors`.
They are kept for reference only and are **not maintained**.

- `standalone_detectors/`: early single-file prototypes (MediaPipe on a webcam or video, no depth).
  Pass a video path as the first argument, or nothing to use webcam 0.
- `demos/`: per-exercise RealSense demos and validation scripts. Most of them import classes that
  no longer exist (`PendulumSwingTracker`, `ShoulderCircleTracker`, `SitToStandChecklist`,
  `get_exercise_components`), so they will not run without porting them to the current detector API.

Use `code/app.py` for the maintained application.
