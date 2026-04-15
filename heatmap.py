# heatmap.py - Accumulates centroid positions into a spatial heatmap

import numpy as np
import cv2

class DwellHeatmap:
    """
    Accumulates centroid positions frame-by-frame into a heatmap.
    Useful for visualizing where objects spend the most time.
    """

    def __init__(self, width, height):
        self.map = np.zeros((height, width), dtype=np.float32)

    def update(self, track_map):
        for obj_id, (cx, cy) in track_map.items():
            if 0 <= cx < self.map.shape[1] and 0 <= cy < self.map.shape[0]:
                cv2.circle(self.map, (cx, cy), 20, 1.0, -1)

    def render(self, frame, alpha=0.45):
        norm = cv2.normalize(self.map, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(colored, alpha, frame, 1 - alpha, 0)
        return blended

    def save(self, path="outputs/heatmap.png"):
        norm = cv2.normalize(self.map, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(path, colored)
        print(f"[Heatmap] Saved → {path}")