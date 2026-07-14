from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.track(
    source="data/raw/traffic_cam_footage.mp4",
    tracker="configs/bytetrack_custom.yaml",
    save=True,
    conf=0.3,
    vid_stride=5,
    name="track_test",
    persist=True,
)

all_ids = set()
per_frame_counts = []

for frame_index, frame_result in enumerate(results):
    boxes = frame_result.boxes
    if boxes.id is None:
        per_frame_counts.append(0)
        continue
    ids_this_frame = boxes.id.tolist()
    all_ids.update(int(i) for i in ids_this_frame)
    per_frame_counts.append(len(ids_this_frame))

print(f"Unique track IDs seen across whole clip: {len(all_ids)}")
print(f"Average cars visible per frame: {sum(per_frame_counts) / len(per_frame_counts):.1f}")
print(f"Max cars visible in a single frame: {max(per_frame_counts)}")
