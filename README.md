# VectorGuard: Intelligent Speed Detection & Adaptive Traffic Management

An edge AI-driven smart traffic monitoring engine that turns standard CCTV video feeds into real-time traffic intelligence. VectorGuard detects speed infractions, wrong-way driving, illegal halting, U-turns, and reckless zig-zag patterns using deterministic spatial geometry and lightweight YOLOv8 object tracking.

---

## Key Features
- **Unified Single-Feed Analytics**: Concurrent tracking, anomaly detection, and vehicle counting from one stream.
- **Zero Manual Lane Marking**: Self-learns intersection flow vectors during startup and serializes to JSON.
- **Deterministic Edge Geometry**: Evaluates trajectories and speeds using NumPy vector math for sub-50ms latency.
- **Live Dispatch Dashboard**: Streamlit interface displaying real-time chronological infraction tables.

---

## Environment Variables & Configuration
System parameters are configured via `config2.yaml` or optional environment variables:

| Variable / YAML Key | Type | Default | Description |
|---|---|---|---|
| `camera.source` | int/str | `0` | Camera device index (e.g. `0` for webcam) or path to RTSP/MP4 stream |
| `camera.grid_size_pixels` | int | `50` | Grid resolution for spatial vector map discretization |
| `learning.phase_frames` | int | `300` | Number of startup frames to learn baseline directional flow |
| `detection_thresholds.vehicle_classes` | list | `[2, 3, 5, 7]` | COCO classes (car, motorcycle, bus, truck) |
| `speed_estimation.enable` | bool | `true` | Toggles displacement-based velocity calculation |
| `speed_estimation.speed_limit_kmh` | int | `50` | Speed enforcement ceiling |
| `speed_estimation.pixels_per_meter` | float | `8.0` | Camera perspective distance calibration factor |
| `network.enable_webhooks` | bool | `false` | Enables asynchronous cloud telemetry transmission |
| `network.webhook_url` | str | `""` | Target API endpoint for dispatch alerts |

---

## Setup & Installation

### Prerequisites
- Python 3.11 or 3.12 (Recommended for PyTorch / NumPy compatibility)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/yaminibisht299/vectorgaurd-traffic-ai.git
cd vectorguard-traffic-ai