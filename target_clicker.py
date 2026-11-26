import ctypes
#issue is comming with me bcz of scaling
# (without this, a 1920x1080 screen at 125% scaling looks like 1536x864).
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor DPI aware
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()    # fallback for older Windows
    except Exception:
        pass

import time
import cv2
import mss
import numpy as np
import pyautogui
from ultralytics import YOLO

# ======================= SETTINGS =======================
MODEL_PATH = r"runs\detect\runs\detect\target_detection-8\weights\best.pt"

CONF = 0.22
IMGSZ = 640
MONITOR_INDEX = 1         # my 1080p monitor on the left

DEBUG_MODE = False        # False = real clicking, True = move only
PICK_MODE = "confidence"  # "confidence" or "center"
CLICK_COOLDOWN = 0.01     # seconds between clicks

# Remove pyautogui's built-in 0.1s delay and the corner fail-safe
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

VK_ESCAPE = 0x1B
user32 = ctypes.windll.user32


def move_mouse(x, y):
    pyautogui.moveTo(int(x), int(y), duration=0)


def left_click():
    pyautogui.mouseDown(button="left")
    pyautogui.mouseUp(button="left")


def esc_pressed():
    return (user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000) != 0


def pick_target(xywh, confs, frame_w, frame_h):
    """Return the index of the detection to click."""
    if PICK_MODE == "center":
        dist = (xywh[:, 0] - frame_w / 2) ** 2 + (xywh[:, 1] - frame_h / 2) ** 2
        return int(dist.argmin())
    return int(confs.argmax())


def main():
    model = YOLO(MODEL_PATH)

    with mss.mss() as sct:
        monitor = sct.monitors[MONITOR_INDEX]

        print("Using monitor:", monitor)
        print("DEBUG_MODE:", DEBUG_MODE)
        print("Running... press ESC to stop.\n")

        last_click = 0.0

        while not esc_pressed():
            # Capture screen -> BGR frame (frame coords start at monitor's top-left for me -1920 ig)
            frame = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
            frame_h, frame_w = frame.shape[:2]

            boxes = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0].boxes
            if len(boxes) == 0:
                continue

            xywh = boxes.xywh.cpu().numpy()    # [center_x, center_y, w, h] in frame pixels
            confs = boxes.conf.cpu().numpy()

            i = pick_target(xywh, confs, frame_w, frame_h)

            # Frame coords -> absolute desktop coords (monitor may start at negative x)
            target_x = monitor["left"] + int(xywh[i, 0])
            target_y = monitor["top"] + int(xywh[i, 1])

            if DEBUG_MODE:
                move_mouse(target_x, target_y)
            elif time.monotonic() - last_click > CLICK_COOLDOWN:
                move_mouse(target_x, target_y)
                left_click()
                last_click = time.monotonic()
                print(f"CLICK at {target_x}, {target_y} | conf: {confs[i]:.2f}")

    print("Stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")