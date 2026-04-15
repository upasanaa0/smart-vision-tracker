# analytics.py - Detection analytics, logging, and report generation

import json
import csv
import os
from datetime import datetime
from collections import defaultdict


class AnalyticsEngine:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.frame_logs   = []
        self.class_totals = defaultdict(int)
        self.class_peak   = defaultdict(int)
        self.unique_ids   = defaultdict(set)
        self.fps_log      = []
        self.start_time   = datetime.now().isoformat()

    def log_frame(self, frame_idx, detections, fps, tracked_ids=None):
        self.fps_log.append(round(fps, 2))

        for label, count in detections.items():
            self.class_totals[label] += count
            self.class_peak[label] = max(self.class_peak[label], count)

        if tracked_ids:
            for label, ids in tracked_ids.items():
                self.unique_ids[label].update(ids)

        self.frame_logs.append({
            "frame": frame_idx,
            "fps": round(fps, 2),
            "detections": dict(detections)
        })

    def get_summary(self):
        avg_fps = round(sum(self.fps_log) / len(self.fps_log), 2) if self.fps_log else 0
        return {
            "total_frames":          len(self.frame_logs),
            "average_fps":           avg_fps,
            "peak_fps":              round(max(self.fps_log), 2) if self.fps_log else 0,
            "class_totals":          dict(self.class_totals),
            "class_peak_per_frame":  dict(self.class_peak),
            "unique_ids_tracked":    {k: len(v) for k, v in self.unique_ids.items()},
            "start_time":            self.start_time,
            "end_time":              datetime.now().isoformat()
        }

    def export_json(self, filename="analytics.json"):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump({"summary": self.get_summary(), "frame_logs": self.frame_logs}, f, indent=2)
        print(f"[Analytics] JSON saved      → {path}")
        return path

    def export_csv(self, filename="frame_log.csv"):
        path = os.path.join(self.output_dir, filename)
        all_classes = sorted({cls for frame in self.frame_logs for cls in frame["detections"]})
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "fps"] + all_classes)
            for frame in self.frame_logs:
                writer.writerow(
                    [frame["frame"], frame["fps"]] +
                    [frame["detections"].get(cls, 0) for cls in all_classes]
                )
        print(f"[Analytics] CSV saved       → {path}")
        return path

    def export_dashboard_data(self, filename="dashboard_data.json"):
        """
        Exports compact JSON consumed by dashboard.html.
        Keys: summary, classes, timeline
        Timeline entries: {f, fps, <class>: count, ...}
        Downsampled to max 300 points so the chart stays snappy.
        """
        summary = self.get_summary()
        classes = sorted({cls for frame in self.frame_logs for cls in frame["detections"]})

        step = max(1, len(self.frame_logs) // 300)
        timeline = []
        for frame in self.frame_logs[::step]:
            entry = {"f": frame["frame"], "fps": frame["fps"]}
            for cls in classes:
                entry[cls] = frame["detections"].get(cls, 0)
            timeline.append(entry)

        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump({"summary": summary, "classes": classes, "timeline": timeline}, f)
        print(f"[Analytics] Dashboard data  → {path}")
        return path

    def print_summary(self):
        s = self.get_summary()
        print("\n" + "=" * 55)
        print("   SMART VISION TRACKER — ANALYTICS SUMMARY")
        print("=" * 55)
        print(f"  Total Frames   : {s['total_frames']}")
        print(f"  Avg FPS        : {s['average_fps']}")
        print(f"  Peak FPS       : {s['peak_fps']}")
        print(f"  Classes Found  : {list(s['class_totals'].keys())}")
        print(f"\n  Detection Totals:")
        for cls, total in s["class_totals"].items():
            peak   = s["class_peak_per_frame"].get(cls, 0)
            unique = s["unique_ids_tracked"].get(cls, "N/A")
            print(f"    {cls:<15} total={total:<6} peak/frame={peak:<4} unique_ids={unique}")
        print("=" * 55 + "\n")