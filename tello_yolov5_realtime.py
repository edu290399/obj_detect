import argparse
import sys
import time
from contextlib import contextmanager
from typing import List, Tuple

import cv2
import numpy as np
import torch

# Drone control
try:
    from djitellopy import Tello
except Exception as exc:  # pragma: no cover
    Tello = None  # type: ignore
    _import_error = exc
else:
    _import_error = None


def select_device() -> torch.device:
    """Select CUDA if available, else CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_yolov5_model(weights: str = "yolov5s"):
    """
    Load a YOLOv5 model via torch.hub and place it on the optimal device.

    Args:
        weights: One of {'yolov5n','yolov5s','yolov5m','yolov5l','yolov5x'} or path to a .pt file.
    """
    device = select_device()

    # When a local .pt path is supplied, use the 'custom' loader
    if isinstance(weights, str) and weights.lower().endswith(".pt"):
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

    # Reasonable defaults (can be overridden at call site)
    model.conf = 0.25
    model.iou = 0.45
    model.max_det = 200
    return model


def detect_objects(model, frame_bgr: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.45) -> Tuple[List[List[int]], List[str], List[float]]:
    """Run YOLO inference and return boxes, labels, and confidence scores.

    Returns:
        boxes: [x1,y1,x2,y2]
        labels: class names
        scores: confidences
    """
    assert isinstance(frame_bgr, np.ndarray) and frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3

    model.conf = float(conf_thres)
    model.iou = float(iou_thres)
    with torch.no_grad():
        results = model(frame_bgr)

    det = results.xyxy[0].detach().cpu().numpy()  # x1,y1,x2,y2,conf,cls
    h, w = frame_bgr.shape[:2]

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
    h = abs(hash(label))
    return (32 + (h % 200), 32 + ((h // 200) % 200), 32 + ((h // 40000) % 200))


def draw_detections(image_bgr: np.ndarray, boxes: List[List[int]], labels: List[str], scores: List[float]) -> np.ndarray:
    """Overlay bounding boxes and labels on the frame."""
    output = image_bgr.copy()
    for (x1, y1, x2, y2), label, score in zip(boxes, labels, scores):
        color = _color_for_label(label)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        caption = f"{label} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        th = th + baseline
        cv2.rectangle(output, (x1, max(0, y1 - th - 2)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(output, caption, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


@contextmanager
def ensure_safe_land(drone: "Tello"):
    """Context manager to guarantee the drone lands on exit, even on error."""
    try:
        yield
    finally:
        try:
            if drone is not None:
                drone.streamoff()
        except Exception:
            pass
        try:
            if drone is not None:
                drone.land()
                time.sleep(1.0)
        except Exception:
            # Ignore landing errors if already landed / disconnected
            pass


def tello_connect() -> "Tello":
    if Tello is None:
        raise ImportError(f"djitellopy is required but failed to import: {_import_error}")

    drone = Tello()
    drone.connect()
    # Optional: ensure sufficient battery
    try:
        battery = drone.get_battery()
        if isinstance(battery, (int, float)) and battery < 20:
            print(f"Warning: Low battery {battery}% — consider charging before flight.")
    except Exception:
        pass
    return drone


def run_tello_detection(weights: str = "yolov5s", conf_thres: float = 0.25, iou_thres: float = 0.45):
    """Connect to Tello, take off, run real-time detection, then land safely."""
    model = load_yolov5_model(weights)

    drone = tello_connect()
    
    # Prepare video stream with more robust initialization
    print("Initializing video stream...")
    drone.streamoff()
    time.sleep(2.0)
    
    # Try multiple video stream approaches
    frame_source = None
    max_attempts = 5
    
    for attempt in range(max_attempts):
        try:
            print(f"Video stream attempt {attempt + 1}/{max_attempts}")
            
            if attempt == 0:
                # Method 1: Standard djitellopy frame reader
                drone.streamon()
                time.sleep(3.0)
                frame_read = drone.get_frame_read()
                time.sleep(2.0)
                
                if frame_read.frame is not None:
                    print("Standard frame reader successful!")
                    frame_source = frame_read
                    break
                    
            elif attempt == 1:
                # Method 2: Direct UDP video capture
                print("Trying direct UDP video capture...")
                cap = cv2.VideoCapture("udp://0.0.0.0:11111")
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print("Direct UDP capture successful!")
                        frame_source = cap
                        break
                    cap.release()
                    
            elif attempt == 2:
                # Method 3: Try with longer timeout
                print("Trying with extended timeout...")
                drone.streamon()
                time.sleep(5.0)
                frame_read = drone.get_frame_read()
                time.sleep(3.0)
                
                if frame_read.frame is not None:
                    print("Extended timeout successful!")
                    frame_source = frame_read
                    break
                    
            elif attempt == 3:
                # Method 4: Restart stream completely
                print("Restarting video stream...")
                drone.streamoff()
                time.sleep(3.0)
                drone.streamon()
                time.sleep(4.0)
                frame_read = drone.get_frame_read()
                time.sleep(2.0)
                
                if frame_read.frame is not None:
                    print("Restart successful!")
                    frame_source = frame_read
                    break
                    
            else:
                # Method 5: Last resort - try without video
                print("Video stream failed, continuing without video...")
                frame_source = None
                break
                
        except Exception as e:
            print(f"Video stream attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                time.sleep(3.0)
                continue
            else:
                print("All video methods failed, continuing without video...")
                frame_source = None
                break
    
    if frame_source is None:
        print("Warning: No video stream available. Drone will fly without video feed.")
        print("You can still control the drone and see detection results in console.")
    
    with ensure_safe_land(drone):
        # Basic safe flight sequence
        print("Taking off...")
        drone.takeoff()
        time.sleep(2.0)
        try:
            print("Rising 20cm...")
            drone.move_up(20)  # 20 cm
            time.sleep(1.0)
        except Exception as e:
            print(f"Move up failed: {e}")
            pass

        print("Starting object detection...")
        window_name = "Tello YOLOv5 - Press 'q' to land"
        prev_time = time.time()
        frame_count = 0
        
        while True:
            try:
                # Get frame based on source type
                frame = None
                if frame_source is not None:
                    if hasattr(frame_source, 'frame'):  # djitellopy frame reader
                        frame = frame_source.frame
                    elif hasattr(frame_source, 'read'):  # OpenCV capture
                        ret, frame = frame_source.read()
                        if not ret:
                            frame = None
                
                if frame is None:
                    if frame_source is not None:
                        print("No frame received, waiting...")
                        time.sleep(0.1)
                        continue
                    else:
                        # No video - simulate frame for detection demo
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "No Video Feed", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames
                    print(f"Processing frame {frame_count}")

                boxes, labels, scores = detect_objects(model, frame, conf_thres, iou_thres)
                vis = draw_detections(frame, boxes, labels, scores)

                # FPS overlay
                now = time.time()
                fps = 1.0 / max(1e-6, (now - prev_time))
                prev_time = now
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(vis, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Show detection info
                if boxes:
                    cv2.putText(vis, f"Detected: {len(boxes)} objects", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    for i, (label, score) in enumerate(zip(labels, scores)):
                        print(f"Frame {frame_count}: {label} (confidence: {score:.2f})")

                cv2.imshow(window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Landing...")
                    break
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                time.sleep(0.1)
                continue

        cv2.destroyAllWindows()
        
        # Clean up video source
        if frame_source is not None and hasattr(frame_source, 'release'):
            frame_source.release()


def main():
    parser = argparse.ArgumentParser(description="Ryze Tello (TLW004) realtime YOLOv5 detection")
    parser.add_argument("--weights", type=str, default="yolov5s", help="YOLOv5 weights: name or .pt path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    args = parser.parse_args()

    print("Controls: \n- The drone will take off, rise 20 cm, and hover.\n- Press 'q' in the video window to land and exit.")
    try:
        run_tello_detection(args.weights, args.conf, args.iou)
    except KeyboardInterrupt:
        print("Interrupted by user. Landing...")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()


