import json
from pathlib import Path

import cv2

ZONES_FILE = Path("configs/zones.json")


def load_zones(path=ZONES_FILE):
    with open(path) as f:
        config = json.load(f)
    return {name: tuple(box) for name, box in config["zones"].items()}


ZONES = load_zones()

if __name__ == "__main__":
    frame = cv2.imread("data/processed/clean_frame.png")

    for name, (x1, y1, x2, y2) in ZONES.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(frame, name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imwrite("data/processed/zones_preview.png", frame)
