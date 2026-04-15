import cv2
import time
import os
import argparse

from config import CONFIG
from detector import ObjectDetector
from tracker import CentroidTracker
from analytics import AnalyticsEngine

from zone_counter import ZoneCrossingCounter
from heatmap import DwellHeatmap


def run_pipeline(video_source):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    detector = ObjectDetector()
    tracker = CentroidTracker(max_disappeared=25, max_distance=75)
    analytics = AnalyticsEngine(output_dir=CONFIG["output_dir"])

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise IOError(f"Cannot open source: {video_source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps_src = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    counter = ZoneCrossingCounter(
        line_start=(width // 4, height // 2),
        line_end=(3 * width // 4, height // 2)
    )

    heatmap = DwellHeatmap(width, height)

    out_path = os.path.join(CONFIG["output_dir"], "final_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_src, (width, height))

    print(f"[Pipeline] Source: {width}x{height} @ {fps_src}fps")
    if total_frames > 0:
        print(f"[Pipeline] Total frames: {total_frames}")
    print("[Pipeline] Starting detection...\n")

    prev_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, detections, raw_boxes = detector.detect(frame)

        rects = [(x1, y1, x2, y2) for (x1, y1, x2, y2, *_) in raw_boxes]
        track_map = tracker.update(rects)

        count_in, count_out = counter.update(track_map)
        annotated = counter.draw(annotated)

        heatmap.update(track_map)

        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        annotated = detector.draw_hud(annotated, fps, detections, track_map)

        analytics.log_frame(frame_idx, detections, fps)

        writer.write(annotated)

        cv2.imshow("Smart Vision Tracker", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[Pipeline] Stopped by user.")
            break

        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"Frame {frame_idx} | FPS: {fps:.1f} | Objects: {detections} | IN: {count_in} OUT: {count_out}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    heatmap.save()

    analytics.export_json()
    analytics.export_csv()
    analytics.export_dashboard_data()
    analytics.print_summary()

    print(f"\n[Done] Output saved to: {out_path}")
    return out_path, analytics.get_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Vision Tracker")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam (device 0)")
    args = parser.parse_args()

    if args.webcam:
        source = 0
    elif args.video:
        source = args.video
    else:
        print("Usage: python main.py --video input.mp4")
        print("       python main.py --webcam")
        exit(1)

    run_pipeline(source)