# Dataset

Recordings are not stored in git (`*.mp4`, `*.db3` and `*.bag` are ignored) to keep the repository small.

The validation tools in `code/tools/` look for RealSense recordings (`.db3`) in this folder by default.
To use a different folder, set `MIRAZE_DATASET`:

```bash
MIRAZE_DATASET=/path/to/recordings python code/tools/validatesquatangle.py
```

File naming used during data collection: `<exercise>_<camera angle>_<subject>_<take>.db3`,
for example `mini squat_45_manas_1.db3`.
