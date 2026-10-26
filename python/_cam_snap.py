"""每个 DSHOW index 抓一帧存图,用于人工确认设备身份。"""
from pathlib import Path

import cv2

out = Path(r"E:\jaka\SDK v2.2.2\02 example\python\_cam_snapshots")
out.mkdir(exist_ok=True)

for idx in range(5):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[{idx}] 无法打开")
        continue
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame = None
        for _ in range(15):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        if frame is None:
            print(f"[{idx}] 无帧")
            continue
        path = out / f"idx{idx}_{frame.shape[1]}x{frame.shape[0]}.png"
        cv2.imwrite(str(path), frame)
        print(f"[{idx}] {frame.shape[1]}x{frame.shape[0]} -> {path.name}")
    finally:
        cap.release()
