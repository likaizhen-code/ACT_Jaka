"""枚举 DSHOW 相机设备 0-4，探测默认分辨率、MJPG+640x480 是否可设。"""
import cv2

for idx in range(5):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"[{idx}] 无法打开")
        continue

    try:
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        cc = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
        print(f"[{idx}] 默认 {w:.0f}x{h:.0f}  fourcc={cc}")

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, img = cap.read()
        after = f"{img.shape[1]}x{img.shape[0]}" if ok else "读取失败"
        print(f"[{idx}] MJPG+640x480 后实际: {after}")

        if ok and img.shape[:2] != (480, 640):
            # 再试一次：先读几帧再设分辨率（有些驱动要出图后才能切格式）
            for _ in range(3):
                cap.read()
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok2, img2 = cap.read()
            if ok2:
                print(f"[{idx}] 二次设置后实际: {img2.shape[1]}x{img2.shape[0]}")
    finally:
        cap.release()
