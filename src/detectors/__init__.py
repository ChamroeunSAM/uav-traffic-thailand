"""Detector wrappers behind a uniform interface.

Modules:
    base                  -- DetectorWrapper ABC and Detection dataclass.
    ultralytics_wrapper   -- single wrapper covering YOLO11 / YOLO12 / YOLO26
                             (Ultralytics auto-detects the architecture from
                             the model_name).
    rfdetr_wrapper        -- wrapper for the rfdetr package's RFDETR* variants
                             (transformer-based detector, separate API).
    yolo11_sahi_wrapper   -- placeholder for Stage 2 (YOLOv11 + SAHI tiled
                             inference); not used in Stage 1.
"""
