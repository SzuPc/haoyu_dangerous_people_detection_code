import argparse
import gc
import os
import os.path as osp
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
sys.path.append("/home/lenovo/zrl_science_project/dangerous_detection/ai/haoyu_dangerous_people_detection/sam2_main") # /home/lenovo/zrl_science_project/dangerous_detection/ai/samurai/sam2

# =========================
# 路径配置（按你的环境直接写死）
# =========================
YOLO_MODEL_PATH = "/home/lenovo/zrl_science_project/dangerous_detection/ai/haoyu_dangerous_people_detection/checkpoints/yolo_weights/best.pt"
SAMURAI_MODEL_PATH = "/home/lenovo/zrl_science_project/dangerous_detection/ai/haoyu_dangerous_people_detection/checkpoints/sam2.1_hiera_base_plus.pt"
SAMURAI_ROOT = "/home/lenovo/zrl_science_project/dangerous_detection/ai/haoyu_dangerous_people_detection/sam2_main"

sys.path.append(SAMURAI_ROOT)
from sam2.build_sam import build_sam2_video_predictor

# =========================
# 可视化配置
# =========================
MASK_COLOR = (255, 0, 0)
BOX_COLOR = (255, 0, 0)
LABEL_TEXT = "dangerous person"


# =========================
# 基础工具
# =========================
def compute_iou_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return max(0, xB - xA) * max(0, yB - yA)


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))]


def clip_box_xywh(box, width, height):
    x, y, w, h = box
    x = max(0.0, min(x, width - 1))
    y = max(0.0, min(y, height - 1))
    w = max(1.0, min(w, width - x))
    h = max(1.0, min(h, height - y))
    return [x, y, w, h]


def determine_model_cfg(model_path):
    lower = model_path.lower()
    if "large" in lower:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in lower or "b+" in lower:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in lower:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in lower:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    raise ValueError(f"无法根据模型路径判断 SAMURAI 配置: {model_path}")


def mask_to_bbox(mask):
    non_zero = np.argwhere(mask)
    if len(non_zero) == 0:
        return None
    y_min, x_min = non_zero.min(axis=0).tolist()
    y_max, x_max = non_zero.max(axis=0).tolist()
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


# =========================
# 卡尔曼滤波器：状态 [cx, cy, w, h, vx, vy, vw, vh]
# =========================
class BBoxKalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.initialized = False

    @staticmethod
    def xywh_to_measurement(box):
        x, y, w, h = box
        cx = x + w / 2.0
        cy = y + h / 2.0
        return np.array([[cx], [cy], [w], [h]], dtype=np.float32)

    @staticmethod
    def state_to_xywh(state):
        cx, cy, w, h = state[:4].reshape(-1).tolist()
        x = cx - w / 2.0
        y = cy - h / 2.0
        return [x, y, w, h]

    def init(self, box):
        meas = self.xywh_to_measurement(box)
        self.kf.statePost = np.array([
            [meas[0, 0]], [meas[1, 0]], [meas[2, 0]], [meas[3, 0]],
            [0.0], [0.0], [0.0], [0.0]
        ], dtype=np.float32)
        self.initialized = True

    def update(self, measured_box=None):
        if not self.initialized:
            if measured_box is None:
                raise ValueError("KalmanFilter 尚未初始化，且当前没有测量框。")
            self.init(measured_box)
            return measured_box

        pred = self.kf.predict()
        pred_box = self.state_to_xywh(pred)

        if measured_box is None:
            return pred_box

        corrected = self.kf.correct(self.xywh_to_measurement(measured_box))
        return self.state_to_xywh(corrected)


# =========================
# 阶段1：检测首次触发帧
# =========================
def find_trigger_frame_and_box(video_path, yolo_model_path, conf_threshold=0.1, imgsz=640):
    model = YOLO(yolo_model_path)
    model.fuse()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    trigger_frame_idx = None
    matched_box_xywh = None

    frame_idx = 0
    print("🔍 开始检测首次危险目标触发帧...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=imgsz, conf=conf_threshold, verbose=False)[0]
        boxes_tensor = results.boxes.xyxy
        if boxes_tensor is None or boxes_tensor.numel() == 0:
            frame_idx += 1
            continue

        boxes = boxes_tensor.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        idx_0 = np.where((classes == 0) & (confs > 0.75))[0]   # person
        idx_1 = np.where(classes == 1)[0]                       # weapon
        idx_2 = np.where((classes == 2) & (confs > 0.75))[0]   # knife

        def hit_pair(idx_a, idx_b):
            for i in idx_a:
                for j in idx_b:
                    box_a, box_b = boxes[i], boxes[j]
                    inter = compute_iou_area(box_a, box_b)
                    b_area = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
                    if inter > 0.5 * b_area:
                        return box_a if classes[i] in (0, 2) else box_b
            return None

        hit_box = hit_pair(idx_0, idx_1)
        if hit_box is None:
            hit_box = hit_pair(idx_1, idx_2)

        if hit_box is not None:
            trigger_frame_idx = frame_idx
            matched_box_xywh = xyxy_to_xywh(hit_box)
            print(f"🚨 首次匹配成功: frame={trigger_frame_idx}, box={matched_box_xywh}")
            break

        frame_idx += 1

    cap.release()

    return {
        "trigger_found": trigger_frame_idx is not None,
        "trigger_frame_idx": trigger_frame_idx,
        "matched_box_xywh": matched_box_xywh,
        "fps": fps,
        "width": width,
        "height": height,
    }


# =========================
# 阶段2：将触发帧后的部分单独导出给 SAMURAI
# =========================
def export_part2_video(video_path, trigger_frame_idx, output_part2_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    Path(output_part2_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_part2_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= trigger_frame_idx:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


# =========================
# 阶段3：SAMURAI 跟踪 + Kalman 平滑
# =========================
def run_samurai_tracking_with_kalman(part2_video_path, init_box_xywh, samurai_model_path):
    model_cfg = determine_model_cfg(samurai_model_path)
    predictor = build_sam2_video_predictor(model_cfg, samurai_model_path, device="cuda:0")

    cap = cv2.VideoCapture(part2_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开 part2 视频: {part2_video_path}")

    loaded_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        loaded_frames.append(frame)
    cap.release()

    if len(loaded_frames) == 0:
        raise ValueError("part2 视频没有任何帧。")

    height, width = loaded_frames[0].shape[:2]
    init_box_xywh = clip_box_xywh(init_box_xywh, width, height)
    x, y, w, h = init_box_xywh
    init_box_xyxy = [x, y, x + w, y + h]

    kf = BBoxKalmanFilter()
    tracked_results = []

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(part2_video_path, offload_video_to_cpu=True)
        predictor.add_new_points_or_box(state, box=init_box_xyxy, frame_idx=0, obj_id=0)

        expected_idx = 0
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            while expected_idx < frame_idx:
                # 如果 predictor 跳帧，补一个纯预测
                pred_box = clip_box_xywh(kf.update(None), width, height)
                tracked_results.append({
                    "frame_idx": expected_idx,
                    "raw_box": None,
                    "smooth_box": pred_box,
                    "mask": None,
                })
                expected_idx += 1

            current_mask = None
            raw_box = None
            for obj_id, mask in zip(object_ids, masks):
                if int(obj_id) != 0:
                    continue
                mask_np = mask[0].detach().cpu().numpy() > 0.0
                current_mask = mask_np
                raw_box = mask_to_bbox(mask_np)
                break

            if raw_box is None:
                smooth_box = clip_box_xywh(kf.update(None), width, height)
            else:
                raw_box = clip_box_xywh(raw_box, width, height)
                smooth_box = clip_box_xywh(kf.update(raw_box), width, height)

            tracked_results.append({
                "frame_idx": frame_idx,
                "raw_box": raw_box,
                "smooth_box": smooth_box,
                "mask": current_mask,
            })
            expected_idx = frame_idx + 1

    # 对尾部没补齐的帧做预测
    while len(tracked_results) < len(loaded_frames):
        pred_box = clip_box_xywh(kf.update(None), width, height)
        tracked_results.append({
            "frame_idx": len(tracked_results),
            "raw_box": None,
            "smooth_box": pred_box,
            "mask": None,
        })

    del predictor, state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.clear_autocast_cache()
        except Exception:
            pass

    return loaded_frames, tracked_results


# =========================
# 阶段4：输出最终视频
# =========================
def draw_label(frame, x1, y1, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    y_text = max(y1 - 6, th + 2)
    cv2.putText(frame, text, (x1, y_text), font, font_scale, color, thickness, lineType=cv2.LINE_AA)


def write_final_video(
    input_video_path,
    output_video_path,
    trigger_frame_idx,
    tracked_frames,
    tracked_results,
    draw_pre_trigger_annotated=False,
    yolo_model_path=None,
    conf_threshold=0.1,
    imgsz=640,
):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开输入视频: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    yolo_model = None
    if draw_pre_trigger_annotated:
        yolo_model = YOLO(yolo_model_path)
        yolo_model.fuse()

    frame_idx = 0
    track_map = {item["frame_idx"]: item for item in tracked_results}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < trigger_frame_idx:
            if draw_pre_trigger_annotated and yolo_model is not None:
                results = yolo_model.predict(frame, imgsz=imgsz, conf=conf_threshold, verbose=False)[0]
                frame = results.plot()
            out.write(frame)
        else:
            local_idx = frame_idx - trigger_frame_idx
            vis = tracked_frames[local_idx].copy()
            item = track_map.get(local_idx)

            if item is not None and item["mask"] is not None:
                mask_img = np.zeros((height, width, 3), dtype=np.uint8)
                mask_img[item["mask"]] = MASK_COLOR
                vis = cv2.addWeighted(vis, 1.0, mask_img, 0.2, 0)

            if item is not None:
                x, y, w, h = map(int, map(round, item["smooth_box"]))
                x2 = min(width - 1, x + w)
                y2 = min(height - 1, y + h)
                cv2.rectangle(vis, (x, y), (x2, y2), BOX_COLOR, 2)
                draw_label(vis, x, y, LABEL_TEXT, BOX_COLOR)

            out.write(vis)

        frame_idx += 1

    cap.release()
    out.release()


# =========================
# 总流程
# =========================
def process_video(
    video_path,
    output_video_path,
    conf_threshold=0.1,
    imgsz=640,
    draw_pre_trigger_annotated=False,
):
    info = find_trigger_frame_and_box(
        video_path=video_path,
        yolo_model_path=YOLO_MODEL_PATH,
        conf_threshold=conf_threshold,
        imgsz=imgsz,
    )

    if not info["trigger_found"]:
        raise RuntimeError("整段视频没有检测到满足条件的危险目标组合，未生成输出视频。")

    with tempfile.TemporaryDirectory(prefix="samurai_tmp_") as tmp_dir:
        part2_video_path = osp.join(tmp_dir, "part2.mp4")
        export_part2_video(video_path, info["trigger_frame_idx"], part2_video_path)

        tracked_frames, tracked_results = run_samurai_tracking_with_kalman(
            part2_video_path=part2_video_path,
            init_box_xywh=info["matched_box_xywh"],
            samurai_model_path=SAMURAI_MODEL_PATH,
        )

        write_final_video(
            input_video_path=video_path,
            output_video_path=output_video_path,
            trigger_frame_idx=info["trigger_frame_idx"],
            tracked_frames=tracked_frames,
            tracked_results=tracked_results,
            draw_pre_trigger_annotated=draw_pre_trigger_annotated,
            yolo_model_path=YOLO_MODEL_PATH,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
        )

    print("✅ 处理完成")
    print(f"   输入视频: {video_path}")
    print(f"   输出视频: {output_video_path}")
    print(f"   首次触发帧: {info['trigger_frame_idx']}")
    print(f"   初始框: {info['matched_box_xywh']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="输入视频路径")
    parser.add_argument(
        "--output_video_path",
        type=str,
        default="/home/lenovo/zrl_science_project/dangerous_detection/ai/samurai/output/final_detect_samurai_kalman.mp4",
        help="输出视频路径",
    )
    parser.add_argument("--conf_threshold", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--draw_pre_trigger_annotated",
        action="store_true",
        help="触发前是否保留 YOLO 标注结果；默认关闭，即触发前输出原视频帧",
    )
    args = parser.parse_args()

    process_video(
        video_path=args.video_path,
        output_video_path=args.output_video_path,
        conf_threshold=args.conf_threshold,
        imgsz=args.imgsz,
        draw_pre_trigger_annotated=args.draw_pre_trigger_annotated,
    )


"""
python scripts/detect_samurai_kalman_all_in_one.py \
  --video_path /home/lenovo/zrl_science_project/dangerous_detection/ai/samurai/6月23日_refine.mp4 \
  --output_video_path /home/lenovo/zrl_science_project/dangerous_detection/ai/samurai/result.mp4

"""