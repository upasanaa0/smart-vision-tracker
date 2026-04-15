# config.py - Central configuration for Smart Vision Tracker

CONFIG = {
    "model": "yolov8n.pt",
    "confidence": 0.4,
    "iou_threshold": 0.5,
    "target_classes": None,
    "output_dir": "outputs",
    "input_video": None,
    "use_webcam": False,
    "display_fps": True,
    "display_counts": True,
    "display_tracks": True,
    "line_thickness": 2,
    "font_scale": 0.65,
    "colors": {
        "box": (0, 255, 100),
        "text": (255, 255, 255),
        "fps": (0, 200, 255),
        "count": (255, 220, 0),
        "track": (255, 100, 200),
    }
}