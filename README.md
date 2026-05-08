# UAV Traffic Thailand

Vehicle detection and tracking from UAV (drone) traffic video footage captured over
Thailand road networks. The project benchmarks modern detectors (YOLOv9, YOLOv10,
YOLO11, YOLO12, YOLO26, RT-DETR; plus YOLO11+SAHI in Stage 2) and trackers
(ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT)
across multiple flight altitudes (90 m and 120 m) and produces evaluated vehicle
trajectories suitable for downstream traffic-engineering analysis.

## Setup

TODO: setup instructions (Python >=3.10, environment, data download).

## Methodology

See [`docs/`](docs/) for the annotation guide, calibration protocol, and decision log.
