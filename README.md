## Results

| Video | Resolution | Model | Avg FPS | mAP (COCO) |
|-------|-----------|-------|---------|------------|
| sample_street.mp4 | 1280×720 | YOLOv8n | 28.4 | 37.3 |
| sample_mall.mp4 | 640×480 | YOLOv8n | 41.2 | 37.3 |

## Architecture

Upload → FastAPI → YOLOv8 Detector → Centroid Tracker
                                    → Zone Counter (tripwire)
                                    → Dwell Heatmap
                                    → Analytics Engine → JSON/CSV/Dashboard