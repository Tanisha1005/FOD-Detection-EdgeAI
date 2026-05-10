"""
live_detect_mqtt.py — Live webcam detection → MQTT → Dashboard
──────────────────────────────────────────────────────────────
Captures laptop webcam, runs YOLOv10s, publishes to dashboard.

Usage:
    1. Start Mosquitto broker (in another terminal)
    2. Start dashboard.py (in another terminal)
    3. Run this script
"""

import base64
import json
import time
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# ─── CONFIG ──────────────────────────────────────────────
MODEL_PATH   = "best.pt"        # Your trained model
BROKER       = "localhost"
PORT         = 1883
CAMERA_ID    = "camera_01"
SOURCE       = 0                # 0 = default webcam
CONF_THRESH  = 0.35
PUBLISH_HZ   = 4                # publishes per second
# ─────────────────────────────────────────────────────────

# Risk taxonomy — all 31 FOD-A classes
FOD_CLASSES = {
    "bolt":"CRITICAL","nut":"CRITICAL","screw":"CRITICAL",
    "washer":"CRITICAL","nail":"CRITICAL",
    "adjustable_wrench":"CRITICAL","hammer":"CRITICAL",
    "pliers":"CRITICAL","screwdriver":"CRITICAL",
    "wrench":"CRITICAL","cutter":"CRITICAL",
    "battery":"HIGH","wire":"HIGH","tape":"HIGH",
    "concrete":"MODERATE","rock":"MODERATE",
    "metal_part":"MODERATE","metal_sheet":"MODERATE","asphalt":"MODERATE",
    "bottle":"LOW","can":"LOW","gloves":"LOW","plastic_part":"LOW",
    "rubber":"LOW","wood":"LOW","paper":"LOW","shoe":"LOW",
    "luggage_tag":"LOW","label":"LOW","headphone":"LOW","foil":"LOW",
}

RISK_BGR = {
    "CRITICAL": (0, 0, 220),
    "HIGH":     (0, 140, 255),
    "MODERATE": (0, 210, 210),
    "LOW":      (40, 200, 40),
}


def annotate_frame(frame, detections):
    """Draw bounding boxes and labels on the frame."""
    h, w = frame.shape[:2]
    for det in detections:
        xc, yc, bw, bh = det["bbox_norm"]
        x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
        x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
        color = RISK_BGR.get(det["risk"], (180, 180, 180))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} {det['confidence']:.2f} [{det['risk']}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(frame, label, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    ts = time.strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, f"CAM-01 | {ts}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode()


def main():
    print("[Sentinel] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print(f"[Sentinel] Loaded {len(model.names)} classes")

    # MQTT setup
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    base   = f"sentinel/v1/{CAMERA_ID}"
    client.will_set(f"{base}/status", "Offline", retain=True)

    try:
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        client.publish(f"{base}/status", "Online", retain=True)
        print(f"[Sentinel] Connected to MQTT broker at {BROKER}:{PORT}")
    except Exception as e:
        print(f"MQTT connection failed: {e}")
        print("Is Mosquitto running? Start it first.")
        return

    # Camera setup
    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("[Sentinel] Webcam opened. Publishing detections...")
    print("[Sentinel] Press Ctrl+C to stop.")

    publish_interval = 1.0 / PUBLISH_HZ
    cycle    = 0
    start_t  = time.time()
    last_pub = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            now = time.time()
            if now - last_pub < publish_interval:
                continue
            last_pub = now
            cycle += 1

            # YOLO inference
            results = model(frame, conf=CONF_THRESH, verbose=False)[0]

            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_nm = model.names[cls_id]
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                w, h = frame.shape[1], frame.shape[0]
                xc   = ((x1 + x2) / 2) / w
                yc   = ((y1 + y2) / 2) / h
                bw   = (x2 - x1) / w
                bh   = (y2 - y1) / h

                xm = round((xc - 0.5) * 15.0, 2)
                ym = round((1 - yc) * 50.0 + 5.0, 2)

                detections.append({
                    "class":        cls_nm,
                    "confidence":   round(conf, 3),
                    "bbox_norm":    [round(xc, 3), round(yc, 3),
                                     round(bw, 3), round(bh, 3)],
                    "world_coords": [xm, ym],
                    "risk":         FOD_CLASSES.get(cls_nm.lower(), "LOW"),
                })

            # Build full telemetry
            telemetry = {
                "timestamp":  time.time(),
                "camera_id":  CAMERA_ID,
                "cycle":      cycle,
                "detections": detections,
                "motion": {
                    "triggered": len(detections) > 0,
                    "zone":      "Zone-A (0-20m)" if detections else None,
                },
                "radar": {
                    "sweep_angle": (cycle * 7) % 360,
                    "range_m":     100,
                    "returns": [
                        {
                            "distance_m": d["world_coords"][1],
                            "angle_deg":  90 + d["world_coords"][0] * 3,
                            "intensity":  d["confidence"],
                        }
                        for d in detections
                    ],
                },
                "health": {
                    "cpu_temp_c":    round(50 + np.random.uniform(-5, 10), 1),
                    "cpu_usage_pct": round(np.random.uniform(30, 70), 1),
                    "ram_used_mb":   round(np.random.uniform(400, 550), 1),
                    "inference_fps": round(PUBLISH_HZ, 1),
                    "uptime_s":      round(time.time() - start_t, 1),
                },
            }

            # Publish telemetry JSON
            client.publish(f"{base}/telemetry", json.dumps(telemetry))

            # Publish annotated camera frame
            frame_b64 = annotate_frame(frame.copy(), detections)
            client.publish(f"{base}/image", frame_b64)

            # Console log
            if detections:
                names_and_conf = [(d["class"], f"{d['confidence']*100:.0f}%") for d in detections]
                print(f"[{cycle:04d}] DETECTED: {names_and_conf}")
            else:
                print(f"[{cycle:04d}] No detections")

    except KeyboardInterrupt:
        print("\n[Sentinel] Shutting down...")
    finally:
        client.publish(f"{base}/status", "Offline", retain=True)
        client.loop_stop()
        client.disconnect()
        cap.release()
        print("[Sentinel] Done.")


if __name__ == "__main__":
    main()