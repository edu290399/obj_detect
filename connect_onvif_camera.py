import os
import time
from typing import List, Optional

import cv2
from onvif import ONVIFCamera
from urllib.parse import urlparse, urlunparse, quote


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

    prev = time.time()
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

            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev))
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("ASECAM Stream", frame)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


