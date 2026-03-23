# 🚑 Dangerous People Detection with Stable Tracking

## 📌 Overview

We present a practical pipeline for **dangerous human detection and stable tracking in surgical/complex scenes**, combining:

- **YOLO-based detection** for reliable event triggering  
- **SAMURAI tracking** for mask-level temporal consistency  
- **Kalman filtering** for trajectory smoothing and robustness  

This two-stage framework enables **robust, low-jitter tracking after detection**, making it suitable for real-world deployment scenarios such as medical or safety-critical environments.

---

## 🧠 Method Pipeline

Our pipeline consists of two stages:

### Stage 1: Detection (YOLO)

- A YOLO model is applied frame-by-frame.
- Once a **target object (dangerous person)** is detected with sufficient confidence,  
  a **trigger frame** is determined.

### Stage 2: Tracking + Optimization

From the trigger frame onward:

1. **SAMURAI** performs mask-based tracking  
2. Bounding boxes are extracted from masks  
3. A **Kalman Filter** is applied to:
   - smooth trajectories  
   - handle temporary tracking failures  

---

## 📂 Dataset & Weights

All datasets and pretrained weights are available at:

👉 https://huggingface.co/datasets/HPro12/dangerous_people_detection

### Contents

- YOLO training dataset (images + labels)
- Pretrained YOLO weights
- Example videos
- (Optional) SAMURAI-related checkpoints

---

## ⚙️ Installation

```bash
git clone https://github.com/your_repo/dangerous_people_detection.git
cd dangerous_people_detection

conda create -n dangerous python=3.10 -y
conda activate dangerous

pip install -r requirements.txt
```

### Key Dependencies

- PyTorch
- ultralytics (YOLO)
- OpenCV
- numpy

---

## 🏋️ YOLO Training

We use **Ultralytics YOLO** for detection.

### 1. Dataset Structure

```
dataset/
 ├── images/
 │    ├── train/
 │    └── val/
 ├── labels/
 │    ├── train/
 │    └── val/
 └── data.yaml
```

### 2. Train YOLO

```bash
yolo detect train \
  data=dataset/data.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16
```

### 3. Output

Trained weights will be saved to:

```
runs/detect/train/weights/best.pt
```

---

## 🎬 Inference (Detection + Tracking + Kalman)

We provide a unified script:

```
detect_samurai_kalman_all_in_one.py
```

### Run

```bash
python detect_samurai_kalman_all_in_one.py \
  --video_path input.mp4 \
  --output_video_path output.mp4
```

### Optional

```bash
--draw_pre_trigger_annotated
```

- Enables visualization of YOLO detections before trigger

---

## 📤 Output

The output video:

- Same resolution as input
- Same FPS as input
- Encoded with `mp4v`
- Includes:
  - detection boxes (before trigger)
  - segmentation masks (after trigger)
  - Kalman-smoothed bounding boxes

---

## 🔧 Key Features

- ✅ Two-stage detection → tracking pipeline  
- ✅ Mask-level tracking (not just boxes)  
- ✅ Kalman-based temporal smoothing  
- ✅ Robust to occlusion and short-term tracking loss  
- ✅ Easy deployment on real videos  

---

## 📊 Applications

- Surgical scene analysis  
- Safety monitoring  
- Human-object interaction tracking  
- Medical AI pipelines  

---

## 📎 Citation (Coming Soon)

```bibtex
@article{your_project_2026,
  title={Dangerous People Detection with Stable Tracking},
  author={Your Name},
  year={2026}
}
```

---

## ✉️ Contact

If you have any questions, feel free to reach out.
