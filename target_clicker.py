import ctypes
import time
import cv2
import mss
import numpy as np
from ultralytics import YOLO

# ======================= SETTINGS =======================
MODEL_PATH = r"runs\detect\runs\detect\target_detection-8\weights\best.pt"

CONF = 0.22
IMGSZ = 640
MONITOR_INDEX = 1         # your 1080p monitor on the left

DEBUG_MODE = False        # False = real clicking, True = move only
PICK_MODE = "confidence"  # "confidence" or "center"
CLICK_COOLDOWN = 0.01     # seconds between clicks
CLICK_HOLD = 0.01         # how long the button is held down
MOVE_SETTLE = 0.01        # pause between moving and clicking
# ========================================================

user32 = ctypes.windll.user32

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
VK_ESCAPE = 0x1B


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("u", _INPUT_UNION)]


def _send_mouse(flags):
    inp = INPUT(type=0, u=_INPUT_UNION(mi=MOUSEINPUT(0, 0, 0, flags, 0, 0)))
    return user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def move_mouse(x, y):
    user32.SetCursorPos(int(x), int(y))


def left_click():
    _send_mouse(MOUSEEVENTF_LEFTDOWN)
    time.sleep(CLICK_HOLD)
    _send_mouse(MOUSEEVENTF_LEFTUP)


def esc_pressed():
    return (user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000) != 0


model = YOLO(MODEL_PATH)

sct = mss.MSS()
monitor = sct.monitors[MONITOR_INDEX]

print("Using monitor:", monitor)
print("DEBUG_MODE:", DEBUG_MODE)
print("Running... press ESC to stop.\n")

last_click = 0.0

try:
    while True:
        if esc_pressed():
            print("ESC pressed, stopping.")
            break

        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        frame_h, frame_w = frame.shape[:2]

        results = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        cxs = (xyxy[:, 0] + xyxy[:, 2]) / 2
        cys = (xyxy[:, 1] + xyxy[:, 3]) / 2

        if PICK_MODE == "center":
            dist = (cxs - frame_w / 2) ** 2 + (cys - frame_h / 2) ** 2
            i = int(dist.argmin())
        else:
            i = int(confs.argmax())

        target_x = monitor["left"] + int(cxs[i])
        target_y = monitor["top"] + int(cys[i])

        now = time.time()

        if DEBUG_MODE:
            move_mouse(target_x, target_y)
        elif now - last_click > CLICK_COOLDOWN:
            move_mouse(target_x, target_y)
            time.sleep(MOVE_SETTLE)
            left_click()
            last_click = time.time()
            print(f"CLICK at {target_x}, {target_y} | conf: {confs[i]:.2f}")

finally:
    sct.close()