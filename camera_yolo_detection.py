import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from onvif import ONVIFCamera
from urllib.parse import urlparse, urlunparse, quote
from dotenv import load_dotenv

# Import camera utilities
from camera_utils import (
    setup_camera_display,
    ensure_full_frame_visible,
    add_camera_info_overlay,
    draw_motion_contours
)

# Import motion detection
from motion_detector import create_motion_detector

# Load environment variables
load_dotenv('camera_config.env')


def select_device() -> torch.device:
    """Select CUDA if available, else CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_yolov5_model(weights: str = None):
    """
    Load a YOLOv5 model via torch.hub.
    
    Args:
        weights: One of {'yolov5n','yolov5s','yolov5m','yolov5l','yolov5x'} or path to a .pt file.
    
    Returns:
        model: The loaded YOLOv5 model with device and precision set.
    """
    device = select_device()
    
    # Load model name from environment or use default
    weights = weights or os.getenv('YOLO_MODEL', 'yolov5s')
    
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


def discover_rtsp_uri(host: str, port: int, username: str, password: str) -> Optional[str]:
    """
    Use ONVIF to fetch the RTSP stream URI. Returns None if discovery fails.
    """
    try:
        cam = ONVIFCamera(host, port, username, password)
        media = cam.create_media_service()
        profiles = media.GetProfiles()
        if not profiles:
            return None
        
        req = media.create_type("GetStreamUri")
        req.ProfileToken = profiles[0].token
        req.StreamSetup = {"Stream": "RTP-Unicast", "Transport": {"Protocol": "RTSP"}}
        uri = media.GetStreamUri(req).Uri
        
        if not uri:
            return None
        
        # Inject credentials into the RTSP URL
        parsed = urlparse(uri)
        hostname = parsed.hostname or host
        port_rtsp = parsed.port or 554
        netloc = f"{quote(username)}:{quote(password)}@{hostname}:{port_rtsp}"
        return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        return None


def candidate_rtsp_uris(host: str, username: str, password: str) -> List[str]:
    """Common RTSP paths for ONVIF/H.264/H.265 cameras to try as fallbacks."""
    creds = f"{quote(username)}:{quote(password)}"
    return [
        f"rtsp://{creds}@{host}:554/Streaming/Channels/101",  # main stream
        f"rtsp://{creds}@{host}:554/Streaming/Channels/102",  # sub stream
        f"rtsp://{creds}@{host}:554/h264/ch1/main/av_stream",
        f"rtsp://{creds}@{host}:554/h264/ch1/sub/av_stream",
        f"rtsp://{creds}@{host}:554/live/ch00_0",
        f"rtsp://{creds}@{host}:554/live/ch00_1",
    ]


def open_stream(rtsp_url: str, timeout_ms: int = 5000) -> Optional[cv2.VideoCapture]:
    """Try to open RTSP stream with OpenCV; returns VideoCapture or None."""
    # Improve connection reliability for some cameras
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;" + str(timeout_ms * 1000))
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


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


def run_camera_detection(
    model,
    host: str = None,
    port: int = None,
    username: str = None,
    password: str = None,
    conf_thres: float = None,
    iou_thres: float = None,
    motion_cooldown: float = 10.0,
):
    """Connect to ONVIF camera, perform real-time object detection, and display results."""
    
    # Load configuration from environment variables with fallbacks
    host = host or os.getenv('CAMERA_HOST', '192.168.1.10')
    port = port or int(os.getenv('CAMERA_PORT', '80'))
    username = username or os.getenv('CAMERA_USERNAME', 'admin')
    password = password or os.getenv('CAMERA_PASSWORD', '140781')
    conf_thres = conf_thres or float(os.getenv('CONFIDENCE_THRESHOLD', '0.25'))
    iou_thres = iou_thres or float(os.getenv('IOU_THRESHOLD', '0.45'))
    
    print(f"Connecting to camera at {host}:{port}")
    print(f"Using credentials: {username}:***")
    print(f"Detection settings: conf={conf_thres}, iou={iou_thres}")
    print(f"Motion detection: cooldown={motion_cooldown}s")
    
    # Initialize motion detector
    motion_detector = create_motion_detector(cooldown_seconds=motion_cooldown)
    print(f"Motion detector initialized. Saves to: {motion_detector.save_directory}")
    
    print("Discovering RTSP URL via ONVIF...")
    rtsp_url = discover_rtsp_uri(host, port, username, password)
    
    if not rtsp_url:
        print("ONVIF discovery failed. Trying fallback RTSP paths...")
        for url in candidate_rtsp_uris(host, username, password):
            print(f"Trying {url}")
            cap = open_stream(url)
            if cap is not None:
                rtsp_url = url
                cap.release()
                break
    
    if not rtsp_url:
        print("Failed to resolve an RTSP URL. Please verify camera settings/network.")
        return
    
    print(f"Using RTSP URL: {rtsp_url}")
    
    cap = open_stream(rtsp_url)
    if cap is None:
        print("Could not open stream with OpenCV.")
        return
    
    # Setup camera display using utility functions
    window_name = "ASECAM Camera - YOLOv5 Detection + Motion"
    frame_width, frame_height, fps_camera = setup_camera_display(cap, window_name)
    
    prev_time = time.time()
    frame_count = 0
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; reconnecting...")
                cap.release()
                cap = open_stream(rtsp_url)
                if cap is None:
                    print("Reconnect failed.")
                    break
                continue
            
            # Ensure we're working with the full frame
            frame = ensure_full_frame_visible(frame, (frame_width, frame_height))
            
            # Process motion detection
            motion_detected, saved_image_path = motion_detector.process_frame(frame)
            
            # Get motion status for display
            motion_status = "Motion Detected!" if motion_detected else "No Motion"
            motion_cooldown_status = motion_detector.get_cooldown_status()
            
            # Perform object detection on the full frame
            boxes, labels, scores = infer_image(model, frame, conf_thres, iou_thres)
            
            # Draw detections on the full frame
            vis = draw_detections(frame, boxes, labels, scores)
            
            # Draw motion contours if motion was detected
            if motion_detected:
                _, motion_mask, contours = motion_detector.detect_motion(frame)
                vis = draw_motion_contours(vis, contours, motion_detected)
                
                # Show capture status
                if saved_image_path:
                    print(f"📸 Motion capture saved: {saved_image_path}")
                else:
                    print(f"⏰ Motion detected but in cooldown. {motion_cooldown_status} remaining.")
            
            # FPS and detection count overlay
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now
            frame_count += 1
            
            # Add comprehensive camera information overlay
            vis = add_camera_info_overlay(
                vis, 
                host, 
                (frame_width, frame_height), 
                fps, 
                len(boxes),
                f"Frame: {frame_count}",
                motion_status,
                motion_cooldown_status
            )
            
            # Display the full frame - OpenCV will handle the window sizing
            cv2.imshow(window_name, vis)
            
            # Exit on 'q'
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
                
    finally:
        cap.release()
        motion_detector.cleanup()
        cv2.destroyAllWindows()


def main():
    print("Loading YOLOv5 model...")
    model = load_yolov5_model()
    print("Model loaded successfully!")
    
    print("Starting camera detection with motion capture...")
    print("Press 'q' to quit")
    print("Motion captures will be saved automatically with 10-second cooldown")
    
    run_camera_detection(model, motion_cooldown=10.0)


if __name__ == "__main__":
    main()
