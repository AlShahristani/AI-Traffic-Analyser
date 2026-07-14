from collections import Counter

from ultralytics import YOLO

from define_zones import ZONES

model = YOLO("yolov8n.pt")


def get_zone(cx, cy):
    for name, (x1, y1, x2, y2) in ZONES.items():
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return name
    return None


results = model.track(
    source="data/raw/traffic_cam_footage.mp4",
    tracker="configs/bytetrack_custom.yaml",
    conf=0.3,
    vid_stride=5,
    persist=True,
    verbose=False,
)

track_zones = {}  # track_id -> {"first": zone, "last": zone, "class": name}

for frame_result in results:
    boxes = frame_result.boxes
    if boxes.id is None:
        continue
    for track_id, cls, xyxy in zip(boxes.id.tolist(), boxes.cls.tolist(), boxes.xyxy.tolist()):
        track_id = int(track_id)
        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        zone = get_zone(cx, cy)
        if zone is None:
            continue
        class_name = model.names[int(cls)]
        if track_id not in track_zones:
            track_zones[track_id] = {
                "first": zone,
                "last": zone,
                "class": class_name,
                "frames_seen": 1,
                "path": [zone],
            }
        else:
            track_zones[track_id]["last"] = zone
            track_zones[track_id]["frames_seen"] += 1
            if zone != track_zones[track_id]["path"][-1]:
                track_zones[track_id]["path"].append(zone)

MIN_FRAMES = 4
before_filter = len(track_zones)
track_zones = {tid: info for tid, info in track_zones.items() if info["frames_seen"] >= MIN_FRAMES}
print(f"Dropped {before_filter - len(track_zones)} tracks seen for fewer than {MIN_FRAMES} frames (likely noise/fragmentation)")
print()

routes = Counter()
for info in track_zones.values():
    if info["first"] != info["last"]:
        routes[(info["first"], info["last"], info["class"])] += 1

print(f"Total tracked vehicles that stayed in one zone: {sum(1 for i in track_zones.values() if i['first'] == i['last'])}")
print(f"Total tracked vehicles with a real route: {sum(routes.values())}")
print()
for (start, end, cls), count in routes.most_common():
    print(f"{start} -> {end} ({cls}): {count}")

single_zone = [i for i in track_zones.values() if i["first"] == i["last"]]

print()
print("Single-zone tracks by zone:")
for zone, count in Counter(i["first"] for i in single_zone).most_common():
    print(f"  {zone}: {count}")

print()
print("Single-zone tracks by dwell time (frames tracked):")
dwell_buckets = Counter()
for i in single_zone:
    frames = i["frames_seen"]
    if frames == 1:
        bucket = "1 frame"
    elif frames <= 3:
        bucket = "2-3 frames"
    elif frames <= 10:
        bucket = "4-10 frames"
    else:
        bucket = "11+ frames"
    dwell_buckets[bucket] += 1
for bucket in ["1 frame", "2-3 frames", "4-10 frames", "11+ frames"]:
    print(f"  {bucket}: {dwell_buckets[bucket]}")

print()
print("Path length (distinct zone segments visited) across all kept tracks:")
path_lengths = Counter(len(i["path"]) for i in track_zones.values())
for length, count in sorted(path_lengths.items()):
    print(f"  {length} zone(s): {count}")

multi_hop = [i for i in track_zones.values() if len(i["path"]) >= 3]
if multi_hop:
    print()
    print(f"Tracks visiting 3+ distinct zones ({len(multi_hop)}):")
    for i in multi_hop:
        print(f"  {' -> '.join(i['path'])} ({i['class']}, {i['frames_seen']} frames)")
