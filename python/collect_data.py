import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import msvcrt
except ImportError:
    # msvcrt 仅 Windows 可用，本采集脚本也只支持 Windows
    msvcrt = None

# 本机的 time.perf_counter() 在不同进程中各自从零起算（沙箱隔离了
# QPC），跨进程时间戳不可比较；改用 Windows 精确系统时钟（微秒级，
# 全进程一致，Unix 秒）
import ctypes

_kernel32 = ctypes.WinDLL("kernel32")
_filetime = ctypes.c_ulonglong()


def clock_now():
    _kernel32.GetSystemTimePreciseAsFileTime(
        ctypes.byref(_filetime)
    )
    return _filetime.value / 1e7 - 11644473600.0


# ================= 配置 =================

# 控制器固件低于 1.7.1 时必须用 SDK v2.1.x（v2.2.x 登录会被拒），
# 此 SDK 需要 Python 3.9（jaka39 环境）和 VC++ 2013 运行库
SDK_DIR = r"E:\jaka\JAKA_data_collect\Windows\python3\x64"
ROBOT_IP = "10.5.5.100"

CAMERA_FRONT = 3
CAMERA_WRIST = 1

WIDTH = 640
HEIGHT = 480
CAMERA_FPS = 30

# 每次关节读取之间的等待时间。全量状态接口约 65ms/次（轻量接口
# get_joint_position 登录十几秒后必被控制器断开返回 -3，不可用），
# 关节采样约 13-15Hz，对 10fps 导出网格的插值精度足够
ROBOT_PAUSE = 0.005

# 导出频率：并不代表硬件实际采样频率
EXPORT_FPS = 10

# 时间对齐阈值
CAMERA_TOLERANCE = 0.050     # 相机与目标时间偏差：50 ms
CAMERA_PAIR_TOLERANCE = 0.050
ROBOT_MAX_GAP = 0.150       # 关节插值最大跨度：150 ms
# SDK 单次读取最大耗时：每次采集的首次调用有约 190ms 的冷启动，
# 属正常现象（时间戳取前后中点，误差可控），250ms 以上才算异常
ROBOT_MAX_QUERY = 0.250

SAVE_ROOT = Path(r"E:\jaka\SDK v2.2.2")

# ========================================


def write_json_line(file, data):
    file.write(json.dumps(data) + "\n")
    file.flush()


def capture_worker(name, root, stop, start, heartbeat, messages):
    """每个设备在独立进程中采集，避免 SDK 阻塞其他设备。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    robot = None
    camera = None
    logged_in = False

    try:
        if name == "robot":
            sys.path.insert(0, SDK_DIR)
            dll_handle = os.add_dll_directory(SDK_DIR)

            import jkrc

            robot = jkrc.RC(ROBOT_IP)
            ret = robot.login()

            # 控制器冷启动或连接刚断开时，login 常先返回 -1（连接中），
            # 需要等待后重试才能成功
            attempts = 0
            while ret[0] != 0 and attempts < 10:
                time.sleep(2)
                ret = robot.login()
                attempts += 1

            if ret[0] != 0:
                raise RuntimeError(f"机器人登录失败：{ret}")

            logged_in = True

        else:
            import cv2

            index = CAMERA_FRONT if name == "front" else CAMERA_WRIST

            # 设备刚释放就重新打开时，DSHOW 偶尔不生效分辨率设置，
            # 因此打开后验证实际输出，不对就整个重来
            last_error = None
            for attempt in range(3):
                camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)

                if not camera.isOpened():
                    last_error = f"无法打开相机 {name}，编号 {index}"
                    camera = None
                    time.sleep(1)
                    continue

                camera.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*"MJPG"),
                )
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
                camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

                # 预热：拿到第一帧有效图像即可
                image = None
                for _ in range(10):
                    ok, image = camera.read()
                    if ok and image is not None:
                        break
                    image = None

                if image is None:
                    last_error = f"{name} 相机预热失败（第 {attempt+1} 次）"
                elif image.shape != (HEIGHT, WIDTH, 3):
                    last_error = (
                        f"{name} 实际分辨率 {image.shape}，与配置 "
                        f"{(HEIGHT, WIDTH, 3)} 不一致（第 {attempt+1} 次）"
                    )
                else:
                    last_error = None
                    break

                camera.release()
                camera = None
                time.sleep(1)

            if camera is None:
                raise RuntimeError(
                    f"{last_error}；重试 3 次仍失败，"
                    "可尝试重新插拔相机 USB"
                )

            (root / name).mkdir()

        heartbeat.value = clock_now()
        messages.put(("ready", name, ""))

        # 相机在等待启动期间丢弃积压帧，保证开始后第一帧是当前画面；
        # 机器人定期读一次全量状态：消化首次调用约 190ms 的冷启动，
        # 让正式录制时第一次调用就是正常的 65ms
        next_warmup = 0.0
        while not start.wait(0.5):
            if stop.is_set():
                return

            if name != "robot":
                camera.read()
            elif time.perf_counter() >= next_warmup:
                robot.get_robot_status()
                next_warmup = time.perf_counter() + 2.0

        with (root / f"{name}.jsonl").open(
            "x", encoding="utf-8"
        ) as file:
            count = 0

            while not stop.is_set():
                before = clock_now()

                if name == "robot":
                    # 只有全量状态接口在长时间运行/闲置后依然稳定
                    # （轻量接口登录十几秒后返回 -3，不可用）
                    ret = robot.get_robot_status()
                    after = clock_now()

                    if ret[0] != 0:
                        raise RuntimeError(f"状态读取失败：{ret}")

                    status = ret[1]

                    if len(status) <= 22 or not status[22]:
                        raise RuntimeError("状态格式异常或 SDK 通信断开")

                    joints = np.asarray(status[19], dtype=float)

                    if joints.shape != (6,) or not np.isfinite(joints).all():
                        raise RuntimeError("六轴关节数据异常")

                    record = {
                        "seq": count,
                        # 查询前后中点，不是控制器内部测量时间
                        "t": (before + after) / 2,
                        "before": before,
                        "after": after,
                        "q": joints.tolist(),
                    }

                else:
                    ok, image = camera.read()
                    after = clock_now()

                    if not ok or image is None:
                        raise RuntimeError(f"{name} 相机读取失败")

                    if image.shape != (HEIGHT, WIDTH, 3):
                        raise RuntimeError(f"{name} 相机分辨率发生变化")

                    relative_path = f"{name}/{count:08d}.png"

                    ok, encoded = cv2.imencode(
                        ".png",
                        image,
                        [cv2.IMWRITE_PNG_COMPRESSION, 1],
                    )
                    if not ok:
                        raise RuntimeError("图像编码失败")

                    # 支持中文、空格路径
                    encoded.tofile(str(root / relative_path))

                    record = {
                        "seq": count,
                        # 图像接收时间，不是曝光时间
                        "t": after,
                        "before": before,
                        "after": after,
                        "path": relative_path,
                    }

                write_json_line(file, record)
                count += 1
                heartbeat.value = clock_now()

                if name == "robot":
                    stop.wait(ROBOT_PAUSE)

    except BaseException as exc:
        messages.put(("error", name, repr(exc)))
        stop.set()

    finally:
        if camera is not None:
            camera.release()

        if logged_in:
            robot.logout()


def collect_one(seconds, task):
    """采集一条 episode。返回 (status, root, error, interrupted)。"""
    if CAMERA_FRONT == CAMERA_WRIST:
        raise ValueError("两个相机编号不能相同")

    root = (
        SAVE_ROOT
        / "raw_episodes"
        / datetime.now().strftime("episode_%Y%m%d_%H%M%S_%f")
    )
    root.mkdir(parents=True, exist_ok=False)

    metadata = {
        "status": "incomplete",
        "task": task,
        "width": WIDTH,
        "height": HEIGHT,
        "joint_unit": "rad",
        "clock": "GetSystemTimePreciseAsFileTime, unix seconds",
        "camera_timestamp": "host receive time, not exposure time",
        "robot_timestamp": "query midpoint, not controller sample time",
        "camera_indices": [CAMERA_FRONT, CAMERA_WRIST],
        "sdk": SDK_DIR,
    }

    def save_metadata():
        (root / "episode.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    save_metadata()

    context = mp.get_context("spawn")
    stop = context.Event()
    start = context.Event()
    messages = context.Queue()

    workers = []

    for name in ("robot", "front", "wrist"):
        heartbeat = context.Value("d", clock_now())

        process = context.Process(
            target=capture_worker,
            args=(
                name, root, stop, start, heartbeat, messages
            ),
        )
        process.start()
        workers.append((name, process, heartbeat))

    ready = set()
    started = None
    armed = False
    error = None
    interrupted = False
    boot_time = clock_now()
    last_print = boot_time

    print("保存目录：", root, flush=True)
    print("等待机器人和两个相机准备就绪……", flush=True)

    def drain_keys():
        if msvcrt is not None:
            while msvcrt.kbhit():
                msvcrt.getwch()

    def enter_pressed():
        return (
            msvcrt is not None
            and msvcrt.kbhit()
            and msvcrt.getwch() in ("\r", "\n")
        )

    try:
        while True:
            try:
                kind, name, detail = messages.get(timeout=0.1)

                if kind == "error":
                    raise RuntimeError(f"{name}: {detail}")

                if kind == "ready":
                    ready.add(name)
                    print(f"{name} 已准备好", flush=True)

            except queue.Empty:
                pass

            now = clock_now()

            if len(ready) == 3 and started is None:
                if not armed:
                    armed = True
                    drain_keys()
                    print(
                        "设备就绪。按 Enter 开始采集本条。",
                        flush=True,
                    )
                elif enter_pressed():
                    started = now
                    metadata["record_start"] = now
                    start.set()
                    drain_keys()

                    if seconds > 0:
                        print(
                            f"开始采集，{seconds:.0f} 秒后自动停止，"
                            "也可按 Enter 提前结束。",
                            flush=True,
                        )
                    else:
                        print(
                            "开始采集，按 Enter 结束本条采集。",
                            flush=True,
                        )

            # 初始化超时只约束设备就绪阶段；等用户按 Enter 不计时
            if len(ready) < 3 and now - boot_time > 30:
                raise RuntimeError("设备初始化超过 30 秒")

            for name, process, heartbeat in workers:
                if not process.is_alive():
                    raise RuntimeError(f"{name} 采集进程退出")

                if started is not None:
                    if now - max(heartbeat.value, started) > 5:
                        raise RuntimeError(
                            f"{name} 超过 5 秒未更新，停止本次采集"
                        )

            if started is not None:
                if now - last_print >= 1:
                    print(
                        f"已采集 {now - started:.1f} 秒",
                        flush=True,
                    )
                    last_print = now

                # Enter 正常结束本条 episode，与到时停止一样走 complete 路径
                if enter_pressed():
                    print("\n收到 Enter，结束本条采集。", flush=True)
                    break

                if seconds > 0 and now - started >= seconds:
                    break

    except KeyboardInterrupt:
        interrupted = True
        if started is None:
            error = "设备未准备好就停止"
        print("\n收到 Ctrl+C，停止本条采集。")

    except Exception as exc:
        error = str(exc)

    finally:
        metadata["record_end"] = clock_now()
        stop.set()

        # 卡在 SDK 调用里的 worker 最长约 25 秒会自行返回并执行
        # finally 里的 logout；多等一会儿让它干净退出，避免
        # terminate 留下死会话在控制器上堆积，导致后续采集
        # 越来越容易失败
        for name, process, heartbeat in workers:
            process.join(timeout=30)

            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                error = error or f"{name} 未能正常退出"

            elif process.exitcode != 0:
                error = error or f"{name} 异常退出"

        # 检查停止期间产生的错误
        while True:
            try:
                kind, name, detail = messages.get_nowait()
                if kind == "error":
                    error = error or f"{name}: {detail}"
            except queue.Empty:
                break

        metadata["status"] = "failed" if error else "complete"
        metadata["error"] = error
        save_metadata()

    print("采集结果：", metadata["status"])
    print("原始数据：", root)

    return metadata["status"], root, error, interrupted


def collect(episodes, seconds, task):
    if seconds < 0:
        raise ValueError("采集时长不能为负；0 表示不限时，按 Enter 结束")
    if episodes < 1:
        raise ValueError("episodes 至少为 1")

    results = []
    failed_streak = 0

    for i in range(1, episodes + 1):
        print(f"\n========== 第 {i}/{episodes} 条 ==========", flush=True)

        status, root, error, interrupted = collect_one(seconds, task)
        results.append((status, root))

        if interrupted:
            print("\n收到 Ctrl+C，退出整个采集流程。")
            break

        if status == "complete":
            failed_streak = 0
            if i < episodes:
                print(
                    f"第 {i}/{episodes} 条完成。"
                    "复位机器人和物体，下一条设备就绪后按 Enter 开始。",
                    flush=True,
                )
        else:
            failed_streak += 1
            print(f"第 {i}/{episodes} 条失败：{error}", flush=True)

            if failed_streak >= 2:
                print("连续失败 2 次，设备可能掉线，退出采集。")
                break

            if i < episodes:
                print("将继续尝试下一条（Ctrl+C 可退出）。", flush=True)

    done = sum(1 for s, _ in results if s == "complete")

    print("\n========== 采集汇总 ==========")
    print(f"成功 {done} 条 / 尝试 {len(results)} 条")
    for status, root in results:
        mark = "[OK ]" if status == "complete" else "[失败]"
        print(f" {mark} {root}")
    print("逐条体检：python check_episode.py <episode 目录>")


def read_stream(path):
    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    if len(rows) < 2:
        raise ValueError(f"数据不足：{path}")

    times = np.asarray([row["t"] for row in rows], dtype=float)

    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"时间戳异常：{path}")

    return rows, times


def nearest_indices(times, targets):
    right = np.clip(
        np.searchsorted(times, targets), 0, len(times) - 1
    )
    left = np.maximum(right - 1, 0)

    return np.where(
        np.abs(times[left] - targets)
        <= np.abs(times[right] - targets),
        left,
        right,
    )


def export_episode(root, output):
    # 这些依赖只在导出模式中加载
    import cv2
    from importlib.metadata import version
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        CODEBASE_VERSION,
    )

    if version("lerobot") != "0.4.3" or CODEBASE_VERSION != "v3.0":
        raise RuntimeError("请使用 lerobot==0.4.3 的独立导出环境")

    if output.exists():
        raise FileExistsError("输出目录已存在，请换一个新目录")

    metadata = json.loads(
        (root / "episode.json").read_text(encoding="utf-8")
    )

    if metadata["status"] != "complete":
        raise RuntimeError("该示范采集不完整，不允许直接导出")

    streams = {
        name: read_stream(root / f"{name}.jsonl")
        for name in ("robot", "front", "wrist")
    }

    # 三路数据共同覆盖的时间区间
    begin = max(
        metadata["record_start"],
        *[stream[1][0] for stream in streams.values()],
    )
    end = min(
        metadata["record_end"],
        *[stream[1][-1] for stream in streams.values()],
    )

    frame_count = int(np.floor((end - begin) * EXPORT_FPS)) + 1
    if frame_count < 2:
        raise RuntimeError("共同覆盖的时间不足")

    grid = begin + np.arange(frame_count) / EXPORT_FPS

    # 关节角按前后两次观测线性插值
    robot_rows, robot_times = streams["robot"]
    joints = np.asarray([row["q"] for row in robot_rows])

    if joints.shape != (len(robot_rows), 6):
        raise RuntimeError("关节状态维度异常")
    if not np.isfinite(joints).all():
        raise RuntimeError("关节状态包含无效数值")

    right = np.clip(
        np.searchsorted(robot_times, grid, side="right"),
        1,
        len(robot_times) - 1,
    )
    left = right - 1
    gaps = robot_times[right] - robot_times[left]

    query_times = np.asarray([
        row["after"] - row["before"] for row in robot_rows
    ])

    if np.any(gaps > ROBOT_MAX_GAP):
        raise RuntimeError("机器人反馈间隔过大，拒绝跨越断流插值")

    if np.any(
        np.maximum(query_times[left], query_times[right])
        > ROBOT_MAX_QUERY
    ):
        raise RuntimeError("SDK 调用耗时过大，时间对齐不可靠")

    weights = (grid - robot_times[left]) / gaps
    states = (
        joints[left]
        + weights[:, None] * (joints[right] - joints[left])
    ).astype(np.float32)

    # 拖动示教没有控制命令：以对齐后的下一时刻实测位置近似目标动作。
    # 图像和 observation.state 保持在 t；action 使用 t + 1/fps。
    # 最后一帧没有未来目标，丢弃，不能复制自身或跨 episode 取值。
    aligned_frame_count = frame_count
    actions = states[1:].copy()
    frame_count = aligned_frame_count - 1

    camera_indices = {}
    camera_offsets = {}

    for name in ("front", "wrist"):
        rows, times = streams[name]
        indices = nearest_indices(times, grid)
        offsets = times[indices] - grid

        if np.any(np.abs(offsets) > CAMERA_TOLERANCE):
            raise RuntimeError(f"{name} 相机时间偏差超过阈值")

        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(
                f"{name} 原始帧率不足，降低 EXPORT_FPS 后再导出"
            )

        camera_indices[name] = indices
        camera_offsets[name] = offsets

    pair_offsets = np.abs(
        camera_offsets["front"] - camera_offsets["wrist"]
    )
    if np.any(pair_offsets > CAMERA_PAIR_TOLERANCE):
        raise RuntimeError("两台相机的时间差超过阈值")

    image_shape = (
        metadata["height"], metadata["width"], 3
    )

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "q1_rad", "q2_rad", "q3_rad",
                "q4_rad", "q5_rad", "q6_rad",
            ],
        }
    }

    features["action"] = {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "q1_target_rad", "q2_target_rad", "q3_target_rad",
            "q4_target_rad", "q5_target_rad", "q6_target_rad",
        ],
    }

    for name in ("front", "wrist"):
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        }

    repo_id = "local/jaka_zu3_act"

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        fps=EXPORT_FPS,
        robot_type="jaka_zu3",
        features=features,
        use_videos=True,
        vcodec="h264",
        video_backend="pyav",
    )

    try:
        with (output / "alignment.csv").open(
            "x", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow([
                "frame_index",
                "timestamp",
                "host_time",
                "front_source_index",
                "wrist_source_index",
                "front_offset_s",
                "wrist_offset_s",
                "robot_left_index",
                "robot_right_index",
                "action_host_time",
                "action_robot_left_index",
                "action_robot_right_index",
            ])

            for i, target_time in enumerate(grid[:-1]):
                frame = {
                    "observation.state": states[i].copy(),
                    "action": actions[i].copy(),
                    "task": metadata["task"],
                }

                for name in ("front", "wrist"):
                    row = streams[name][0][camera_indices[name][i]]
                    path = root / row["path"]

                    image = cv2.imdecode(
                        np.fromfile(str(path), dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )

                    if image is None or image.shape != image_shape:
                        raise RuntimeError(f"图像读取失败：{path}")

                    frame[f"observation.images.{name}"] = (
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    )

                # 官方 API 自动生成 timestamp = frame_index / fps
                dataset.add_frame(frame)

                writer.writerow([
                    i,
                    i / EXPORT_FPS,
                    target_time,
                    camera_indices["front"][i],
                    camera_indices["wrist"][i],
                    camera_offsets["front"][i],
                    camera_offsets["wrist"][i],
                    left[i],
                    right[i],
                    grid[i + 1],
                    left[i + 1],
                    right[i + 1],
                ])

        dataset.save_episode()

    finally:
        dataset.finalize()

    report = {
        "source_episode": str(root.resolve()),
        "source_metadata": metadata,
        "frames": frame_count,
        "aligned_frames_before_action_shift": aligned_frame_count,
        "discarded_tail_frames": 1,
        "fps": EXPORT_FPS,
        "front_max_offset_ms": float(
            np.max(np.abs(camera_offsets["front"])) * 1000
        ),
        "wrist_max_offset_ms": float(
            np.max(np.abs(camera_offsets["wrist"])) * 1000
        ),
        "camera_pair_max_offset_ms": float(
            np.max(pair_offsets) * 1000
        ),
        "robot_max_interpolation_gap_ms": float(
            np.max(gaps) * 1000
        ),
        "observation_only": False,
        "action_definition": "next_aligned_measured_joint_position",
        "action_space": "absolute_joint_position_rad",
        "action_is_command_measurement": False,
        "action_horizon_s": 1.0 / EXPORT_FPS,
        "has_gripper": False,
        "hardware_synchronized": False,
    }

    (output / "alignment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 重新加载并检查首尾图像和关节数据
    loaded = LeRobotDataset(
        repo_id=repo_id,
        root=output,
        video_backend="pyav",
    )

    if len(loaded) != frame_count:
        raise RuntimeError("重新加载后的帧数不匹配")

    for index in (0, frame_count - 1):
        sample = loaded[index]

        np.testing.assert_allclose(
            sample["observation.state"].numpy(),
            states[index],
            atol=1e-6,
        )

        np.testing.assert_allclose(
            sample["action"].numpy(), states[index + 1], atol=1e-6
        )

        for name in ("front", "wrist"):
            expected_shape = (
                3, metadata["height"], metadata["width"]
            )
            if tuple(
                sample[f"observation.images.{name}"].shape
            ) != expected_shape:
                raise RuntimeError("视频解码尺寸异常")

    (output / "EXPORT_VALIDATED.json").write_text(
        json.dumps({"frames": frame_count, "version": "v3.0",
                    "has_action": True, "action_horizon_s": 1.0 / EXPORT_FPS}),
        encoding="utf-8",
    )

    print("LeRobot v3.0 导出并验证完成：", output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["collect", "export"]
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="自动停止秒数；0 表示不限时，按 Enter 结束本条采集",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="连续采集的条数，例如 10 表示连采 10 条",
    )
    parser.add_argument("--task", default="抓起物体放入盒子")
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.mode == "collect":
        collect(args.episodes, args.seconds, args.task)
    else:
        if args.episode is None or args.output is None:
            parser.error("导出需要 --episode 和 --output")
        export_episode(args.episode, args.output)
