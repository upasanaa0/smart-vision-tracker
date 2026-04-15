# detector.py - Core YOLOv8 detection + drawing engine

import cv2
import numpy as np
from ultralytics import YOLO
from config import CONFIG


class ObjectDetector:
    """
    Wraps YOLOv8 inference with bounding box drawing,
    label rendering, and optional class filtering.
    """

    def __init__(self):
        self.model = YOLO(CONFIG["model"])
        self.target_classes = CONFIG.get("target_classes")
        self.conf = CONFIG["confidence"]
        self.iou = CONFIG["iou_threshold"]
        self.colors = CONFIG["colors"]
        # Per-class color palette for visual variety
        np.random.seed(42)
        self.class_colors = np.random.randint(80, 255, (80, 3), dtype=np.uint8)

    def detect(self, frame):
        """
        Runs inference on a frame.
        Returns:
            annotated_frame, detections (dict label->count), raw_boxes list of (x1,y1,x2,y2,label,conf)
        """
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
        boxes_data = results[0].boxes

        detections = {}
        raw_boxes = []

        for box in boxes_data:
            cls = int(box.cls[0])
            label = self.model.names[cls]

            if self.target_classes and label not in self.target_classes:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections[label] = detections.get(label, 0) + 1
            raw_boxes.append((x1, y1, x2, y2, label, conf, cls))

        annotated = self._draw(frame.copy(), raw_boxes, detections)
        return annotated, detections, raw_boxes

    def _draw(self, frame, raw_boxes, counts):
        cfg = CONFIG

        for (x1, y1, x2, y2, label, conf, cls) in raw_boxes:
            color = tuple(int(c) for c in self.class_colors[cls % 80])
            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg["line_thickness"])
            # Label background
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, cfg["font_scale"], 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, cfg["font_scale"],
                        (255, 255, 255), 1, cv2.LINE_AA)

        # HUD panel (semi-transparent)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 50 + 30 * max(len(counts), 1)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        return frame

    def draw_hud(self, frame, fps, counts, track_map=None):
        """Draws the FPS + count overlay on top of the frame."""
        if CONFIG["display_fps"]:
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        CONFIG["colors"]["fps"], 2, cv2.LINE_AA)
        if CONFIG["display_counts"]:
            y = 70
            for label, count in counts.items():
                cv2.putText(frame, f"{label}: {count}", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            CONFIG["colors"]["count"], 2, cv2.LINE_AA)
                y += 28

        if CONFIG["display_tracks"] and track_map:
            for obj_id, centroid in track_map.items():
                cx, cy = centroid
                cv2.circle(frame, (cx, cy), 4, CONFIG["colors"]["track"], -1)
                cv2.putText(frame, f"#{obj_id}", (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            CONFIG["colors"]["track"], 1, cv2.LINE_AA)
        return frame
    
# zone_counter.py

class ZoneCrossingCounter:
    """
    Counts objects crossing a virtual line defined by two points.
    Uses the sign of the cross product to detect direction of crossing.
    """

    def __init__(self, line_start, line_end):
        self.p1 = line_start   # (x1, y1)
        self.p2 = line_end     # (x2, y2)
        self.prev_positions = {}   # id -> side (-1 or 1)
        self.count_in = 0
        self.count_out = 0

    def _side(self, point):
        """Returns which side of the line the point is on via cross product."""
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        px = point[0] - self.p1[0]
        py = point[1] - self.p1[1]
        cross = dx * py - dy * px
        return 1 if cross >= 0 else -1

    def update(self, track_map):
        """
        track_map: {obj_id: (cx, cy)} from CentroidTracker
        Returns: (count_in, count_out)
        """
        for obj_id, centroid in track_map.items():
            current_side = self._side(centroid)
            if obj_id in self.prev_positions:
                prev_side = self.prev_positions[obj_id]
                if prev_side != current_side:
                    if current_side == 1:
                        self.count_in += 1
                    else:
                        self.count_out += 1
            self.prev_positions[obj_id] = current_side
        return self.count_in, self.count_out

    def draw(self, frame):
        import cv2
        cv2.line(frame, self.p1, self.p2, (0, 255, 255), 2)
        cv2.putText(frame, f"IN:  {self.count_in}",  (self.p1[0], self.p1[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)
        cv2.putText(frame, f"OUT: {self.count_out}", (self.p1[0], self.p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
        return frame