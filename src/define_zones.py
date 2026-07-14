import cv2

# Rectangles as (x1, y1, x2, y2) in original 3840x2160 pixel coordinates,
# top-left and bottom-right corners of each zone.
ZONES = {
    "north_lanes": (0, 0, 1248, 614),
    "north_queue": (1248, 0, 1920, 518),
    "east_road": (2784, 288, 3840, 1056),
    "south_road": (1152, 1056, 2496, 2160),
}

if __name__ == "__main__":
    frame = cv2.imread("data/processed/clean_frame.png")

    for name, (x1, y1, x2, y2) in ZONES.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(frame, name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imwrite("data/processed/zones_preview.png", frame)
