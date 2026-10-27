import cv2
import numpy as np
from ultralytics import YOLO
import math
import json
import os
import time
import yaml
import threading
import requests 
 
class VectorGuardEngine:
    def __init__(self, config_path="config2.yaml", model_weights="yolov8s.pt"):
        print(f"Loading system configuration from {config_path}...")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
 
        # --- SYSTEM CONFIGURATION ---
        self.GRID_SIZE = config['camera']['grid_size_pixels']
        self.RESIZE_FACTOR = config['camera']['resize_factor']
        self.LEARNING_PHASE_FRAMES = config['learning']['phase_frames']
        self.VEHICLE_CLASSES = config['detection_thresholds']['vehicle_classes']
        
        self.FLOW_MAP_FILE = config['storage']['flow_map_file']
        self.ALERTS_LOG_FILE = config['storage']['alerts_log_file']
 
        # --- NEW: NETWORK CONFIGURATION ---
        self.ENABLE_WEBHOOKS = config['network']['enable_webhooks']
        self.WEBHOOK_URL = config['network']['webhook_url']
        # --- NEW: SPEED ESTIMATION CONFIGURATION ---
        self.ENABLE_SPEED_CHECK = config.get('speed_estimation', {}).get('enable', False)
        self.SPEED_LIMIT_KMH = config.get('speed_estimation', {}).get('speed_limit_kmh', 50)
        self.PIXELS_PER_METER = config.get('speed_estimation', {}).get('pixels_per_meter', 8.0)
        self.vehicle_speed_data = {}  # track_id -> list of (timestamp, cx, cy)
 
        # --- CAMERA SOURCE (CHANGED: was 'video_path', now supports live webcam) ---
        # 'source' can be an integer webcam index (0, 1, 2...) or a string path/URL.
        # Falls back to 0 (default laptop webcam) if not specified, and also
        # supports the old 'video_path' key for backward compatibility.
        raw_source = config['camera'].get('source', config['camera'].get('video_path', 0))
        self.CAMERA_SOURCE = self._resolve_camera_source(raw_source)
 
        # --- STATE TRACKERS ---
        self.flow_map = {} 
        self.frame_count = 0
        self.vehicle_trajectories = {}
        self.vehicle_states = {} 
        self.active_halts = {}   
        self.reported_violations = set() 
 
        # --- INITIALIZE AI ---
        self.model = YOLO(model_weights)
        self.config = config 
        self._load_brain()
 
        # --- STATIC CAMERA INFO ---
        self.CAMERA_ID = config['camera'].get('id', 'CAM_UNKNOWN')
        self.CAMERA_LAT = config['camera'].get('lat', 0.0)
        self.CAMERA_LON = config['camera'].get('lon', 0.0)
 
    @staticmethod
    def _resolve_camera_source(raw_source):
        """Turns a numeric-looking source ('0', 0) into an int webcam index,
        and leaves file paths/URLs (e.g. 'video.mp4', 'rtsp://...') as strings."""
        if isinstance(raw_source, int):
            return raw_source
        if isinstance(raw_source, str) and raw_source.strip().lstrip('-').isdigit():
            return int(raw_source.strip())
        return raw_source
 
    def _send_webhook(self, payload):
        """Background task that handles the internet connection."""
        try:
            # The timeout=2.0 ensures a dead server doesn't keep threads alive forever
            response = requests.post(self.WEBHOOK_URL, json=payload, timeout=2.0)
            if response.status_code == 200:
                print(f"Cloud Alert Success: {payload['violation']} (ID: {payload['track_id']})")
        except Exception as e:
            # We fail silently so network outages don't crash the local computer vision engine
            pass
 
    def _load_brain(self):
        """Loads the learned road vector map from JSON to bypass Phase 1."""
        if os.path.exists(self.FLOW_MAP_FILE):
            print("Found existing brain! Loading learned road map...")
            with open(self.FLOW_MAP_FILE, 'r') as f:
                raw_map = json.load(f)
                # Convert string keys "x,y" back to tuple keys (x, y)
                self.flow_map = {tuple(map(int, k.split(','))): v for k, v in raw_map.items()}
            
            self.frame_count = self.LEARNING_PHASE_FRAMES 
            print("Ready to detect violations immediately.")
        else:
            print("No memory found. Starting Self-Learning Phase from live webcam feed...")
 
    def _save_brain(self):
        """Exports the learned road vector map to JSON."""
        print("\nLearning complete! Exporting brain to JSON...")
        json_ready_map = {f"{k[0]},{k[1]}": v for k, v in self.flow_map.items()}
        with open(self.FLOW_MAP_FILE, 'w') as f:
            json.dump(json_ready_map, f)
        print("Brain saved successfully.")
    
    def _log_alert(self, track_id, violation_type):
        """Appends a detected violation locally and broadcasts it to the cloud."""
        alert_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id": self.CAMERA_ID,
            "track_id": int(track_id),
            "violation": violation_type,
            "lat": self.CAMERA_LAT,
            "lon": self.CAMERA_LON
        }

        # 1. Save Locally (Immediate, blocking write)
        with open(self.ALERTS_LOG_FILE, 'a') as f:
            f.write(json.dumps(alert_data) + "\n")

        # 2. Broadcast to the Cloud (Asynchronous, non-blocking)
        if self.ENABLE_WEBHOOKS and self.WEBHOOK_URL:
            threading.Thread(target=self._send_webhook, args=(alert_data,), daemon=True).start()

    def _calculate_speed_kmh(self, track_id, cx, cy):
        """Estimates real-world speed (km/h) using pixel displacement over
        wall-clock time, converted via the PIXELS_PER_METER calibration factor."""
        now = time.time()

        if track_id not in self.vehicle_speed_data:
            self.vehicle_speed_data[track_id] = []

        self.vehicle_speed_data[track_id].append((now, cx, cy))
        # Keep roughly the last second of samples so the reading stays responsive
        self.vehicle_speed_data[track_id] = self.vehicle_speed_data[track_id][-30:]

        samples = self.vehicle_speed_data[track_id]
        if len(samples) < 2:
            return 0.0

        t_old, x_old, y_old = samples[0]
        t_new, x_new, y_new = samples[-1]

        dt = t_new - t_old
        if dt <= 0:
            return 0.0

        pixel_dist = math.hypot(x_new - x_old, y_new - y_old)
        meters = pixel_dist / self.PIXELS_PER_METER
        speed_kmh = (meters / dt) * 3.6
        return speed_kmh

    def _log_speed_alert(self, track_id, speed_kmh):
        """Same pattern as _log_alert, but includes the measured speed."""
        alert_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id": self.CAMERA_ID,
            "track_id": int(track_id),
            "violation": "OVERSPEEDING",
            "speed_kmh": round(speed_kmh, 1),
            "speed_limit_kmh": self.SPEED_LIMIT_KMH,
            "lat": self.CAMERA_LAT,
            "lon": self.CAMERA_LON
        }

        # 1. Save Locally (Immediate, blocking write)
        with open(self.ALERTS_LOG_FILE, 'a') as f:
            f.write(json.dumps(alert_data) + "\n")
 
        # 2. Broadcast to the Cloud (Asynchronous, non-blocking)
        if self.ENABLE_WEBHOOKS and self.WEBHOOK_URL:
            threading.Thread(target=self._send_webhook, args=(alert_data,), daemon=True).start()
 
    def process_frame(self, frame):
        """Core computer vision and calculus pipeline for a single frame."""
        self.frame_count += 1
        results = self.model.track(frame, persist=True, classes=self.VEHICLE_CLASSES, verbose=False)
        new_trajectories = {}
 
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
 
            # --- CLEAN UP VANISHED VEHICLES ---
            current_ids = set(track_ids)
            self.active_halts = {k: v for k, v in self.active_halts.items() if k in current_ids}
            self.vehicle_states = {k: v for k, v in self.vehicle_states.items() if k in current_ids}
 
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = [int(i) for i in box]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                pt = (cx, cy)
 
                # Update movement history
                if track_id not in self.vehicle_trajectories:
                    new_trajectories[track_id] = [pt]
                else:
                    new_trajectories[track_id] = self.vehicle_trajectories[track_id]
                    new_trajectories[track_id].append(pt)
                
                # Keep last 30 frames for Long-Term Memory
                new_trajectories[track_id] = new_trajectories[track_id][-30:]
                history = new_trajectories[track_id]
                # --- NEW: SPEED CHECK (runs every frame, independent of the learning phase) ---
                speed_kmh = None
                if self.ENABLE_SPEED_CHECK:
                    speed_kmh = self._calculate_speed_kmh(track_id, cx, cy)

                    if speed_kmh >= self.SPEED_LIMIT_KMH:
                        color = (0, 0, 255)
                        label = f"ID: {track_id} - OVERSPEED {int(speed_kmh)}km/h"

                        speed_signature = f"{track_id}_OVERSPEEDING"
                        if speed_signature not in self.reported_violations:
                            self._log_speed_alert(track_id, speed_kmh)
                            self.reported_violations.add(speed_signature)
                    else:
                        label = f"ID: {track_id} - {int(speed_kmh)}km/h"
                        color = (0, 255, 0)
                else:
                    color = (0, 255, 0) # Default Green
                    label = f"ID: {track_id}"
                
                violation_type = None
 
                if len(history) >= 5:
                    dx = history[-1][0] - history[0][0]
                    dy = history[-1][1] - history[0][1]
                    
                    grid_x, grid_y = cx // self.GRID_SIZE, cy // self.GRID_SIZE
                    cell_id = (grid_x, grid_y)
 
                    # --- PHASE 1: LEARNING ---
                    if self.frame_count < self.LEARNING_PHASE_FRAMES:
                        if cell_id not in self.flow_map:
                            self.flow_map[cell_id] = []
                        self.flow_map[cell_id].append((dx, dy))
                        cv2.putText(frame, "LEARNING ROAD...", (10, 30), 0, 0.7, (255, 255, 0), 2)
                        
                        
                    
                    # --- PHASE 2: 4-in-1 ANOMALY DETECTION ---
                    else:
                        # --- PRIORITY 1: ILLEGAL HALTING & QUEUE MANAGEMENT ---
                        if len(history) >= 30:
                            dist_past = math.hypot(history[-15][0] - history[-30][0], history[-15][1] - history[-30][1])
                            dist_recent = math.hypot(history[-1][0] - history[-15][0], history[-1][1] - history[-15][1])
 
                            current_state = self.vehicle_states.get(track_id, "NORMAL")
 
                            if current_state == "NORMAL" and dist_past > 15 and dist_recent < 5:
                                is_queuing = False
                                for halt_id, halt_pos in self.active_halts.items():
                                    if halt_id != track_id:
                                        if math.hypot(cx - halt_pos[0], cy - halt_pos[1]) < 150: 
                                            is_queuing = True
                                            break
 
                                if is_queuing:
                                    self.vehicle_states[track_id] = "QUEUED"
                                else:
                                    self.vehicle_states[track_id] = "HALTED"
                                    self.active_halts[track_id] = (cx, cy)
                                    violation_type = "ILLEGAL_HALT" 
 
                            if current_state == "HALTED":
                                if dist_recent > 10: 
                                    self.vehicle_states[track_id] = "NORMAL"
                                    if track_id in self.active_halts: 
                                        del self.active_halts[track_id]
                                else:
                                    color = (0, 255, 255) 
                                    label = f"ID: {track_id} - HALTED!"
                                    self.active_halts[track_id] = (cx, cy) 
 
                            elif current_state == "QUEUED":
                                if dist_recent > 10: 
                                    self.vehicle_states[track_id] = "NORMAL"
                                else:
                                    color = (255, 255, 0) 
                                    label = f"ID: {track_id} - QUEUED"
 
                        # PRIORITY 2: U-TURN DETECTED
                        if violation_type is None and len(history) >= 15:
                            v_start = (history[5][0] - history[0][0], history[5][1] - history[0][1])
                            v_now = (history[-1][0] - history[-5][0], history[-1][1] - history[-5][1])
                            self_dot = (v_start[0] * v_now[0]) + (v_start[1] * v_now[1])
                            
                            if self_dot < -1000:
                                color = (255, 165, 0) 
                                label = f"ID: {track_id} - U-TURN!"
                                violation_type = "ILLEGAL_U_TURN"
 
                        # PRIORITY 3: WRONG-WAY DRIVING
                        if violation_type is None and cell_id in self.flow_map:
                            # Calculate the total distance the car has moved across its history
                            displacement_total = math.hypot(dx, dy)
                            
                            # NEW: Only check for wrong-way if the car is actively moving (e.g., > 20 pixels)
                            # This ignores crashed cars bouncing backward and YOLO jitter on stalled cars.
                            if displacement_total > 20:
                                avg_dx, avg_dy = np.mean(self.flow_map[cell_id], axis=0)
                                dot_product = (dx * avg_dx) + (dy * avg_dy)
                                
                                if dot_product < -500: 
                                    color = (0, 0, 255) 
                                    label = f"ID: {track_id} - WRONG WAY!"
                                    violation_type = "WRONG_WAY"
                                
                        # PRIORITY 4: ZIG-ZAG / RECKLESS DRIVING
                        if violation_type is None and len(history) >= 15:
                            displacement = math.hypot(history[-1][0] - history[-15][0], history[-1][1] - history[-15][1])
                            
                            # Raise minimum displacement to ignore crawling traffic jams
                            if displacement > 30: 
                                box_w = x2 - x1
                                box_h = y2 - y1
                                box_diagonal = max(math.hypot(box_w, box_h), 1) 
                                
                                dynamic_threshold = 1.15 + (15.0 / box_diagonal)
 
                                # Sub-Sampling: Calculate path length every 3 frames instead of 1
                                # This completely erases YOLO bounding box micro-vibrations
                                path_length = 0
                                step = 3
                                for i in range(len(history) - 14 + step, len(history), step):
                                    path_length += math.hypot(history[i][0] - history[i-step][0], history[i][1] - history[i-step][1])
                                
                                if (path_length / displacement) > dynamic_threshold:
                                    color = (255, 0, 255) 
                                    label = f"ID: {track_id} - ZIG-ZAG!"
                                    violation_type = "RECKLESS_DRIVING" 
 
                        # --- LOGGING ALERTS (Updated with Cooldown) ---
                        if violation_type:
                            # Create a unique signature (e.g., "Car 102 - ZIG-ZAG")
                            violation_signature = f"{track_id}_{violation_type}"
                            
                            # Only fire the webhook if we haven't already reported it
                            if violation_signature not in self.reported_violations:
                                self._log_alert(track_id, violation_type)
                                self.reported_violations.add(violation_signature)
 
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), 0, 0.5, color, 2)
 
        self.vehicle_trajectories = new_trajectories.copy()
        if self.frame_count == self.LEARNING_PHASE_FRAMES - 1:
            self._save_brain()
 
        return frame
 
    def run(self):
        """Starts the main LIVE WEBCAM processing loop."""
        # --- CHANGED: open the live webcam instead of a video file ---
        cap = cv2.VideoCapture(self.CAMERA_SOURCE)
 
        if not cap.isOpened():
            print("Could not open webcam. Check camera permissions or camera index.")
            return
 
        print(f"Webcam opened successfully (source={self.CAMERA_SOURCE}). Press 'q' to quit.")
 
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 30  # ~1 second at 30fps of dropped frames before giving up
 
        try:
            while True:
                ret, frame = cap.read()
 
                if not ret or frame is None:
                    consecutive_failures += 1
                    print("Warning: failed to read frame from webcam. Retrying...")
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print("Too many consecutive failed frame reads. Stopping.")
                        break
                    continue
 
                consecutive_failures = 0
 
                # Pass the live frame into our core engine
                annotated_frame = self.process_frame(frame)
 
                cv2.imshow('VectorGuard - Live Webcam', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
 
 
# ==========================================
# EXECUTING THE ENGINE
# ==========================================
if __name__ == "__main__":
    engine = VectorGuardEngine(config_path='config2.yaml')
    engine.run()