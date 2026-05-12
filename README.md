# ORB-SLAM3 with Dynamic Object Filtering


> **Base system:** ORB-SLAM3 by Campos et al. (University of Zaragoza)  
> **Extensions:** YOLO-based dynamic object filtering · Aerial dataset support · Trajectory evaluation tooling

---

## Table of Contents

1. [What Was Changed from Original ORB-SLAM3](#1-what-was-changed-from-original-orb-slam3)
2. [What Was Added / Built](#2-what-was-added--built)
3. [System Requirements & Dependencies](#3-system-requirements--dependencies)
4. [Build Instructions](#4-build-instructions)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Running the System](#6-running-the-system)
7. [Evaluating Results](#7-evaluating-results)
8. [Camera Configuration Notes](#8-camera-configuration-notes)
9. [Known Limitations](#9-known-limitations)
10. [Reference](#10-reference)

---

## 1. What Was Changed from Original ORB-SLAM3

The following **modifications** were made to existing ORB-SLAM3 source files:

### `CMakeLists.txt`
- Upgraded the required C++ standard from **C++14 → C++17** (required by ONNX Runtime and modern STL features used in the new modules).
- Added `onnxruntime` include/library paths for YOLO inference.
- Registered two new source files: `src/DynamicFilter.cc` and `src/YOLODetector.cc`.

### `src/Tracking.cc` / `include/Tracking.h`
- Added a `DynamicFilter` member object that is initialized alongside the tracker.
- Inserted a call to `DynamicFilter::FilterFrame()` at the start of each frame's feature-processing pipeline, **before** ORB feature matching. This removes dynamic keypoints prior to tracking so they do not corrupt the map.

### `src/DynamicFilter.cc`
- Fixed an `#include "Verbose.h"` error (the `Verbose` class is defined in `System.h`; a separate header did not exist in this build). Changed to `#include "System.h"`.

### `Examples/Monocular/Teknofest.yaml` *(new, not a modification of existing)*
- Created a new configuration YAML for the Teknofest aerial dataset (see [Section 5](#5-dataset-preparation)).

### `Examples/Monocular/TUM3.yaml` *(unchanged)*
- Used as-is for TUM fr3/walking_xyz validation runs.

---

## 2. What Was Added / Built

### `include/YOLODetector.h` · `src/YOLODetector.cc`
A new class that:
- Loads a **YOLOv11-seg** model exported to **ONNX** format.
- Runs inference asynchronously on a background thread to avoid blocking the tracking loop.
- Returns segmentation masks for dynamic object classes (people, vehicles, animals, etc.).
- Supports **CPU and CUDA** execution providers via ONNX Runtime (GPU requires `libonnxruntime_providers_shared.so`).

Key configuration parameters (set in YAML):
```yaml
YOLODetector.ModelPath:     "/path/to/best.onnx"
YOLODetector.InputSize:     320        # Must match ONNX export imgsz
YOLODetector.ConfThreshold: 0.40
YOLODetector.MaskThreshold: 0.50
```

### `include/DynamicFilter.h` · `src/DynamicFilter.cc`
A three-stage cascaded filter applied per frame:

```
Stage 1 – Epipolar check:
    Keypoints with large epipolar residuals (outliers to the fundamental matrix)
    are flagged as potentially dynamic.

Stage 2 – YOLO mask check:
    Keypoints that fall inside a YOLO segmentation mask for a dynamic class
    are removed.

Stage 3 – Optical flow check:
    Keypoints whose optical flow magnitude exceeds the scene median by a
    configurable factor are removed.
```

Only keypoints surviving all three stages are passed to the ORB matcher.


### `evaluation/visualize_results.py`
A rich trajectory evaluation and visualization tool (replaces the bare-bones `evaluate_ate_scale.py`):

**Features:**
- Umeyama alignment with scale (for monocular scale recovery)
- **3D ATE** and **2D ATE (XY-only)** — the latter is the meaningful metric for aerial SLAM where Z is unreliable
- ATE heatmap on the 2D trajectory plot (green = low error → red = high error)
- ATE per-frame time-series (separate panels for 3D and 2D error)
- Start/end markers on trajectory plots
- Statistics summary card (RMSE, Mean, Median, Std, Max, path lengths, scale factor)
---

## 3. System Requirements & Dependencies

| Component | Version / Notes |
|---|---|
| **OS** | Ubuntu 20.04 / 22.04 (or WSL2 on Windows) |
| **CMake** | ≥ 3.10 |
| **GCC / G++** | ≥ 9 (C++17 support required) |
| **OpenCV** | 4.x |
| **Eigen3** | ≥ 3.3 |
| **Pangolin** | For the 3D viewer |
| **ONNX Runtime** | ≥ 1.14 (CPU; CUDA version for GPU acceleration) |
| **Python 3** | For dataset prep and evaluation |
| **Python packages** | `numpy`, `matplotlib`, `scipy` |

Install Python packages:
```bash
pip install numpy matplotlib scipy
```

---

## 4. Build Instructions

```bash
mkdir orb_slam
# Clone or unzip the project
cd /orb_slam

# Build Thirdparty libraries
cd Thirdparty/DBoW2 && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
cd ../../g2o && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
cd ../../..

# Build ORB-SLAM3 (with extensions)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ..
```

> **Note:** If `libonnxruntime_providers_shared.so` is missing, YOLO will fall back to CPU automatically. The SLAM system will still function; only dynamic filtering may be slower.

---

## 5. Dataset Preparation

### TUM RGB-D (for validation)
Download from https://cvg.cit.tum.de/data/datasets/rgbd-dataset

```
datasets/
└── tum/
    └── rgbd_dataset_freiburg3_walking_xyz/
        ├── rgb/
        ├── rgb.txt
        └── groundtruth.txt
```
Download from https://drive.google.com/drive/folders/18_VqLBbyTubVSWAXG_CgmuJWGCx0mcBd 
###  Aerial Dataset
Place raw images and GPS ground truth in:
```
datasets/
└── aerial/
    ├── images/          ← raw frames (3840×2160, 7.5 fps)
    |-- groundtruth.csv  
    ├── groundtruth.txt ---  converted from csv  
```



---

## 6. Running the System

### Run on TUM fr3/walking_xyz (validation)
```bash
# In WSL terminal (from project root)
./Examples/Monocular/mono_tum \
    Vocabulary/ORBvoc.txt \
    Examples/Monocular/TUM3.yaml \
    datasets/tum/rgbd_dataset_freiburg3_walking_xyz
```

### Run on Teknofest Aerial Dataset
```bash
./Examples/Monocular/mono_tum \
    Vocabulary/ORBvoc.txt \
    Examples/Monocular/Teknofest.yaml \
    datasets/aerial
```

On completion, ORB-SLAM3 writes `KeyFrameTrajectory.txt` in the project root.

---

## 7. Evaluating Results

Run the evaluation script from the **Windows** terminal (PowerShell) or WSL:

### Teknofest Aerial
```powershell
# PowerShell
python evaluation/visualize_results.py `
    --est KeyFrameTrajectory.txt `
    --gt  datasets/aerial/groundtruth.txt `
    --dataset "Teknofest Aerial" `
    --out results_teknofest.png
```

### TUM fr3/walking_xyz
```powershell
python evaluation/visualize_results.py `
    --est KeyFrameTrajectory.txt `
    --gt  datasets/tum/rgbd_dataset_freiburg3_walking_xyz/groundtruth.txt `
    --dataset "TUM fr3_walking_xyz" `
    --out results_tum.png
```



### Script Options

| Flag | Default | Description |
|---|---|---|
| `--est` | *(required)* | Path to `KeyFrameTrajectory.txt` |
| `--gt` | *(required)* | Path to `groundtruth.txt` (TUM format) |
| `--dataset` | `"Dataset"` | Label shown in the figure title |
| `--out` | *(none)* | If provided, saves PNG instead of showing interactively |
| `--max_diff` | `0.02` | Max timestamp difference for pose association (seconds) |

### Understanding the Output Figure

| Panel | Description |
|---|---|
| **Left — X/Y/Z vs Time** | Estimated (blue) vs Ground Truth (red dashed). Z row shows raw (faint) and smoothed (solid) estimate. Pink fill = error. |
| **Centre — 2D Trajectory** | Top-down XY view. Estimate coloured by ATE (green→red). Start (▲) and end (■) markers. Colorbar in metres. |
| **Right — 3D Trajectory** | Full 3D path comparison. |
| **Top-right — 3D ATE** | Per-frame absolute trajectory error including Z (inflated for aerial). |
| **Mid-right — 2D ATE** | Per-frame XY-only error — the meaningful metric for aerial. |
| **Bottom-right — Stats Card** | RMSE, Mean, Max for both 3D and 2D ATE; path lengths; scale factor. |

> **Tip for aerial results:** Focus on the **2D ATE RMSE** value. The 3D RMSE is inflated by unavoidable Z oscillations inherent to monocular SLAM on top-down imagery.

---

## 8. Camera Configuration Notes

### Teknofest Aerial — Intrinsics Correction
The camera was calibrated at **4000×3000** pixels (native sensor), but the dataset frames are **3840×2160** (16:9 crop). The principal point must be corrected for the center crop:

```
Horizontal crop offset: (4000 - 3840) / 2 = 80 px
Vertical   crop offset: (3000 - 2160) / 2 = 420 px

cx_corrected = 1988.0 - 80  = 1908.0
cy_corrected = 1562.2 - 420 = 1142.2
```

These corrected values are already applied in `Examples/Monocular/Teknofest.yaml`.

ORB-SLAM3 further downscales frames to **1280×720** (`Camera.newWidth/newHeight`) and divides intrinsics by the scale factor (3×) automatically:

```
After resize: fx=930.7  fy=931.7  cx=636.0  cy=380.7
```

### Key YAML Parameters (Teknofest)
```yaml
Camera1.fx: 2792.2      # Native sensor focal length (pixels at 3840×2160)
Camera1.fy: 2795.2
Camera1.cx: 1908.0      # Corrected for center-crop
Camera1.cy: 1142.2      # Corrected for center-crop

Camera1.k1:  0.0798     # Radial distortion
Camera1.k2: -0.1867

Camera.newWidth:  1280  # Processing resolution (3× downscale)
Camera.newHeight:  720

ORBextractor.nFeatures:  2500
ORBextractor.nLevels:    8
ORBextractor.iniThFAST:  15   # Lowered for flat aerial texture
ORBextractor.minThFAST:   5

System.thFarPoints: 30.0      # Discard map points >30× median depth
```

---

## 9. Known Limitations

| Issue | Root Cause | Mitigation Applied |
|---|---|---|
| **Monocular scale drift** | No absolute scale reference in monocular SLAM | Umeyama alignment with scale in evaluation |
| **~30% frame loss** | Large inter-frame displacement at 7.5 fps | Lowered FAST thresholds; 8-level pyramid |
| **CUDA not available** | `libonnxruntime_providers_shared.so` missing in WSL | YOLO falls back to CPU (slower but functional) |

**True fixes for Z-axis** would require: IMU integration (ORB-SLAM3 IMU mode), stereo camera, or GPS-derived altitude as a Z prior.

---

## 10. Reference

This project is built on top of **ORB-SLAM3**. 

**Official ORB-SLAM3 repository:** https://github.com/UZ-SLAMLab/ORB_SLAM3

---

*This extension was developed for the Teknofest UAV perception challenge. The YOLO model (`best.onnx`) is a custom-trained YOLO11-seg model and is not part of the ORB-SLAM3 distribution.*
