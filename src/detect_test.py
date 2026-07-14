from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# vid_stride=15 samples ~4 frames/sec of real time from this 60fps clip,
# keeping the first CPU test fast enough to iterate on.
results = model.predict(
    source="data/raw/traffic_cam_footage.mp4",
    save=True,
    conf=0.3,
    vid_stride=15,
    name="detect_test",
)
