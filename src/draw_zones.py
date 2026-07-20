import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import simpledialog

import cv2
from PIL import Image, ImageTk

MAX_DISPLAY_SIZE = 1400


def grab_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}")
    height, width = frame.shape[:2]
    return frame, width, height


class ZoneDrawer:
    def __init__(self, root, video_path):
        self.video_path = video_path
        self.zones = {}  # name -> (x1, y1, x2, y2) in ORIGINAL video pixel coords
        self.rect_start = None
        self.live_rect_id = None

        frame_bgr, self.orig_w, self.orig_h = grab_frame(video_path)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        self.scale = min(MAX_DISPLAY_SIZE / self.orig_w, MAX_DISPLAY_SIZE / self.orig_h, 1.0)
        disp_w, disp_h = int(self.orig_w * self.scale), int(self.orig_h * self.scale)

        image = Image.fromarray(frame_rgb).resize((disp_w, disp_h))
        self.photo = ImageTk.PhotoImage(image)

        self.canvas = tk.Canvas(root, width=disp_w, height=disp_h, cursor="cross")
        self.canvas.pack(side=tk.TOP)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        button_bar = tk.Frame(root)
        button_bar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(button_bar, text="Undo Last", command=self.undo_last).pack(side=tk.LEFT)
        tk.Button(button_bar, text="Clear All", command=self.clear_all).pack(side=tk.LEFT)
        tk.Button(button_bar, text="Save & Exit", command=self.save_and_exit).pack(side=tk.LEFT)

        self.root = root

    def on_press(self, event):
        self.rect_start = (event.x, event.y)

    def on_drag(self, event):
        if self.live_rect_id is not None:
            self.canvas.delete(self.live_rect_id)
        x0, y0 = self.rect_start
        self.live_rect_id = self.canvas.create_rectangle(x0, y0, event.x, event.y, outline="lime", width=2)

    def on_release(self, event):
        if self.live_rect_id is not None:
            self.canvas.delete(self.live_rect_id)
            self.live_rect_id = None

        x0, y0 = self.rect_start
        x1, y1 = event.x, event.y
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return  # accidental click/tiny drag, ignore

        name = simpledialog.askstring("Zone name", "Name this zone:", parent=self.root)
        if not name:
            return

        # Canvas coords -> original video pixel coords, and normalize corner order.
        orig_box = (
            round(min(x0, x1) / self.scale),
            round(min(y0, y1) / self.scale),
            round(max(x0, x1) / self.scale),
            round(max(y0, y1) / self.scale),
        )
        self.zones[name] = orig_box
        self.redraw()

    def redraw(self):
        self.canvas.delete("zone")
        for name, (x1, y1, x2, y2) in self.zones.items():
            dx1, dy1, dx2, dy2 = x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale
            self.canvas.create_rectangle(dx1, dy1, dx2, dy2, outline="lime", width=2, tags="zone")
            self.canvas.create_text(dx1 + 5, dy1 + 12, text=name, fill="lime", anchor=tk.NW, tags="zone")

    def undo_last(self):
        if self.zones:
            self.zones.popitem()
            self.redraw()

    def clear_all(self):
        self.zones.clear()
        self.redraw()

    def save_and_exit(self):
        config = {
            "video": self.video_path,
            "frame_width": self.orig_w,
            "frame_height": self.orig_h,
            "zones": {name: list(box) for name, box in self.zones.items()},
        }
        out_path = Path("configs/zones.json")
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved {len(self.zones)} zone(s) to {out_path}")
        self.root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively draw entry/exit zones on a video frame.")
    parser.add_argument("video", nargs="?", default="data/raw/traffic_cam_footage.mp4")
    args = parser.parse_args()

    root = tk.Tk()
    root.title("Draw zones - drag to draw, click Save & Exit when done")
    ZoneDrawer(root, args.video)
    root.mainloop()
