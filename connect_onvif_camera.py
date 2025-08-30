import os
import time
from typing import List, Optional

import cv2
from onvif import ONVIFCamera
from urllib.parse import urlparse, urlunparse, quote

# Import camera utilities
from camera_utils import (
    setup_camera_display,
    ensure_full_frame_visible,
    add_camera_info_overlay,
    draw_motion_contours
)

# Import motion detection
from motion_detector import create_motion_detector


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


def main():
    host = "192.168.1.10"
    port = 80
    username = "admin"
    password = "140781"
    motion_cooldown = 10.0  # 10 second cooldown between captures

    print("Discovering RTSP URL via ONVIF...")
    rtsp_url = discover_rtsp_uri(host, port, username, password)

    if not rtsp_url:
        print("ONVIF discovery failed. Trying fallback RTSP paths...")
        for url in candidate_rtsp_uris(host, username, password):
            print(f"Trying {url}")
            cap = open_stream(url)
            if cap is None:
                continue
            rtsp_url = url
            cap.release()
            break

    if not rtsp_url:
        print("Failed to resolve an RTSP URL. Please verify camera settings/network.")
        return

    print(f"Using RTSP URL: {rtsp_url}")
    print(f"Motion detection: cooldown={motion_cooldown}s")

    cap = open_stream(rtsp_url)
    if cap is None:
        print("Could not open stream with OpenCV.")
        return

    # Initialize motion detector
    motion_detector = create_motion_detector(cooldown_seconds=motion_cooldown)
    print(f"Motion detector initialized. Saves to: {motion_detector.save_directory}")

    # Setup camera display using utility functions
    window_name = "ASECAM Stream + Motion Detection"
    frame_width, frame_height, fps_camera = setup_camera_display(cap, window_name)

    prev = time.time()
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

            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev))
            prev = now
            frame_count += 1
            
            # Draw motion contours if motion was detected
            if motion_detected:
                _, motion_mask, contours = motion_detector.detect_motion(frame)
                frame = draw_motion_contours(frame, contours, motion_detected)
                
                # Show capture status
                if saved_image_path:
                    print(f"📸 Motion capture saved: {saved_image_path}")
                else:
                    print(f"⏰ Motion detected but in cooldown. {motion_cooldown_status} remaining.")
            
            # Add comprehensive camera information overlay
            frame = add_camera_info_overlay(
                frame, 
                host, 
                (frame_width, frame_height), 
                fps, 
                0,  # No detections in this simple viewer
                f"Frame: {frame_count}",
                motion_status,
                motion_cooldown_status
            )
            
            # Display the full frame
            cv2.imshow(window_name, frame)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cap.release()
        motion_detector.cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


