import os
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch


def select_device() -> torch.device:
    """Select CUDA if available, else CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_yolov5_model(weights: str = "yolov5s"):
    """
    Load a YOLOv5 model via torch.hub.

    Args:
        weights: One of {'yolov5n','yolov5s','yolov5m','yolov5l','yolov5x'} or path to a .pt file.

    Returns:
        model: The loaded YOLOv5 model with device and precision set.
    """
    device = select_device()

    # When a local .pt path is supplied, use the 'custom' loader
    if isinstance(weights, str) and weights.lower().endswith(".pt") and os.path.exists(weights):
        model = torch.hub.load("ultralytics/yolov5", "custom", path=weights)
    else:
        # Use pre-trained model from the hub (e.g., 'yolov5s')
        model = torch.hub.load("ultralytics/yolov5", weights, pretrained=True)

    model.to(device)

    # Use half precision on CUDA for speed
    if device.type == "cuda":
        model.half()
    else:
        model.float()

    # Default inference settings (can be adjusted by caller)
    model.conf = 0.25  # confidence threshold
    model.iou = 0.45   # NMS IoU threshold
    model.max_det = 300

    return model


def infer_image(
    model,
    image_bgr: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> Tuple[List[List[int]], List[str], List[float]]:
    """
    Run inference on a single image and return detections.

    Args:
        model: Loaded YOLOv5 model from torch.hub.
        image_bgr: Input image in BGR format (numpy array, HxWx3).
        conf_thres: Confidence threshold for predictions.
        iou_thres: IoU threshold for NMS.

    Returns:
        boxes: List of [x1, y1, x2, y2] integer pixel coords.
        labels: List of class label strings.
        scores: List of confidence scores (floats).
    """
    assert isinstance(image_bgr, np.ndarray) and image_bgr.ndim == 3 and image_bgr.shape[2] == 3, (
        "image_bgr must be an HxWx3 BGR numpy array"
    )

    # Update thresholds per-call if provided
    model.conf = float(conf_thres)
    model.iou = float(iou_thres)

    # For CUDA half-precision models, ensure the input stays as numpy (the model handles conversion)
    with torch.no_grad():
        results = model(image_bgr)

    det = results.xyxy[0].detach().cpu().numpy()  # shape: [num_dets, 6] -> x1,y1,x2,y2,conf,cls
    h, w = image_bgr.shape[:2]

    boxes: List[List[int]] = []
    labels: List[str] = []
    scores: List[float] = []

    for *xyxy, conf, cls in det:
        x1, y1, x2, y2 = xyxy
        x1 = int(max(0, min(w - 1, round(float(x1)))))
        y1 = int(max(0, min(h - 1, round(float(y1)))))
        x2 = int(max(0, min(w - 1, round(float(x2)))))
        y2 = int(max(0, min(h - 1, round(float(y2)))))

        boxes.append([x1, y1, x2, y2])
        class_id = int(cls)
        label = str(results.names.get(class_id, class_id))
        labels.append(label)
        scores.append(float(conf))

    return boxes, labels, scores


def _color_for_label(label: str) -> Tuple[int, int, int]:
    """Create a deterministic BGR color for a given label string."""
    # Simple hash to stable color mapping
    h = abs(hash(label))
    # Ensure reasonably bright and distinct colors
    return (32 + (h % 200), 32 + ((h // 200) % 200), 32 + ((h // 40000) % 200))


def draw_detections(
    image_bgr: np.ndarray,
    boxes: List[List[int]],
    labels: List[str],
    scores: List[float],
) -> np.ndarray:
    """
    Draw bounding boxes with labels and confidence scores on the image.

    Args:
        image_bgr: Source image (BGR).
        boxes: List of [x1, y1, x2, y2].
        labels: List of class names.
        scores: List of confidence scores.

    Returns:
        A copy of the image with drawings.
    """
    output = image_bgr.copy()

    for (x1, y1, x2, y2), label, score in zip(boxes, labels, scores):
        color = _color_for_label(label)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness=2)

        caption = f"{label} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        th = th + baseline
        # Draw filled rectangle for text background
        cv2.rectangle(output, (x1, max(0, y1 - th - 2)), (x1 + tw + 2, y1), color, thickness=-1)
        # Put text (in white)
        cv2.putText(
            output,
            caption,
            (x1 + 1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return output


def run_webcam(
    model,
    cam_index: int = 0,
    window_name: str = "YOLOv5 - Real-time Detection",
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
):
    """Open webcam stream, perform detection, and display results until 'q' is pressed."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam index {cam_index}")

    prev_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            boxes, labels, scores = infer_image(model, frame, conf_thres, iou_thres)
            vis = draw_detections(frame, boxes, labels, scores)

            # FPS overlay
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now
            cv2.putText(
                vis,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, vis)

            # Exit on 'q'
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    # Load model (auto-selects device)
    model = load_yolov5_model("yolov5s")
    # Start webcam detection
    run_webcam(model)


if __name__ == "__main__":
    main()


