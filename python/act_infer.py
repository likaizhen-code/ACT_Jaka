# -*- coding: utf-8 -*-
"""用 lerobot 框架对训练好的 ACT 模型做真机推理（JAKA ZU3）。

机械臂与相机的调用方式与 collect_data.py 一致：
  - jkrc SDK（RC/login 重试/get_robot_status 全量状态取 status[19] 关节）
  - 两个 USB 相机（front=0, wrist=1，640x480 MJPG 30fps，DSHOW 后端）

与采集不同点：相机改为后台线程连续抓帧只保留最新一帧（推理只需要当前
画面，不能让 30fps 的帧在驱动缓冲里积压造成延迟）；采集时是拖动示教，
只读不写，推理要在主循环里发运动指令，因此登录后还要上电、使能。

运行环境：C:\\Users\\SHU\\miniconda3\\envs\\lerobot（Python 3.10，
lerobot 0.4.3 + torch cu126 + opencv）。jkrc 的 pyd 在 3.10 下同样可用，
所以不像采集那样需要 jaka39 环境，单进程即可。

权重说明：checkpoints/1007 由训练机上新版 lerobot 保存，比本地 0.4.3
多一套 policy_preprocessor/policy_postprocessor 归一化管线，且
config.json 里多一个本地不认识的 pretrained_revision 字段。本地
ACTPolicy 内部没有归一化逻辑，因此这里手动完成：
  - 图像：ImageNet mean/std（训练时 use_imagenet_stats=true）
  - 关节状态 / 动作：数据集 MEAN_STD 统计（存于 pre/postprocessor 的
    safetensors 里，eps=1e-8）
权重本身（model.safetensors）与 0.4.3 的 ACTPolicy 结构完全一致，可
strict 加载（已验证）。

用法（在本目录下）：
  C:\\Users\\SHU\\miniconda3\\envs\\lerobot\\python.exe act_infer.py --steps 100

流程：加载策略 → 打开相机 → 登录/上电/使能机器人 → 摆好初始场景按
Enter → 预览首步动作并倒数 3 秒 → 以 10Hz 循环（读关节 → 取帧 →
select_action → 安全限幅 → 非阻塞 joint_move）→ 结束后回到起始关节
位置，退出。--dry-run 只跑相机+推理，不连接机器人、不发运动指令。
"""

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import msvcrt
except ImportError:
    msvcrt = None

# ================= 配置 =================

# 与 collect_data.py 相同的 SDK 路径与机器人地址
SDK_DIR = r"E:\jaka\JAKA_data_collect\Windows\python3\x64"
ROBOT_IP = "10.5.5.100"

# 2026-10-11 实测：0=笔记本内置，1=wrist，2=虚拟摄像头(720p)，3=front。
# 编号随 USB 枚举可能变化，变了先用 cam_check.py 确认（见同目录采集脚本）
CAMERA_FRONT_DEFAULT = 3
CAMERA_WRIST_DEFAULT = 1

WIDTH = 640
HEIGHT = 480
CAMERA_FPS = 30

# 与训练数据导出频率一致的控制周期
CONTROL_FPS = 10

DEFAULT_CKPT = Path(r"E:\jaka\SDK v2.2.2\checkpoints\1012\pretrained_model")

SAVE_ROOT = Path(r"E:\jaka\SDK v2.2.2\rollouts")

# 归一化 eps，与训练时 preprocessor 配置一致
NORM_EPS = 1e-8

# ========================================


def enter_pressed():
    return (
        msvcrt is not None
        and msvcrt.kbhit()
        and msvcrt.getwch() in ("\r", "\n")
    )


def drain_keys():
    if msvcrt is not None:
        while msvcrt.kbhit():
            msvcrt.getwch()


# ---------------- 策略加载 ----------------

def load_policy(ckpt_dir, device, replan_every):
    """加载新版格式的 checkpoint 到本地 lerobot 0.4.3 的 ACTPolicy。

    返回 (policy, norm)，norm 是整理好形状、已搬到 device 的归一化
    统计量。
    """
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from safetensors.torch import load_file, load_model

    ckpt_dir = Path(ckpt_dir)

    raw = json.loads(
        (ckpt_dir / "config.json").read_text(encoding="utf-8")
    )

    if raw.get("type") != "act":
        raise RuntimeError(f"不是 ACT 模型：type={raw.get('type')}")

    # 丢弃本地 ACTConfig 不认识的字段（如新版加的 pretrained_revision）
    valid = {f.name for f in dataclass_fields(ACTConfig)}
    dropped = sorted(k for k in raw if k not in valid and k != "type")
    if dropped:
        print(f"忽略训练环境新增的配置字段：{dropped}")

    def to_feature(d):
        return PolicyFeature(
            type=FeatureType[d["type"]], shape=d["shape"]
        )

    kwargs = {k: v for k, v in raw.items() if k in valid}
    kwargs["input_features"] = {
        k: to_feature(v) for k, v in kwargs["input_features"].items()
    }
    kwargs["output_features"] = {
        k: to_feature(v) for k, v in kwargs["output_features"].items()
    }

    config = ACTConfig(**kwargs)

    # replan_every：每执行多少步用新观测重推一次动作块。checkpoint 里
    # n_action_steps=100（10 秒才重规划）太开环，默认改为每 10 步
    #（1 秒）重规划；0 表示保持 checkpoint 的设置
    if replan_every > 0:
        config.n_action_steps = replan_every

    policy = ACTPolicy(config)
    load_model(policy, str(ckpt_dir / "model.safetensors"), strict=True)
    policy.to(device)
    policy.eval()

    pre_path = (
        ckpt_dir
        / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    )
    post_path = (
        ckpt_dir
        / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )
    for path in (pre_path, post_path):
        if not path.exists():
            raise FileNotFoundError(f"缺少归一化统计文件：{path}")

    pre = load_file(str(pre_path))
    post = load_file(str(post_path))

    def t(name):
        return torch.as_tensor(pre[name]).to(device)

    norm = {
        "state_mean": t("observation.state.mean"),
        "state_std": t("observation.state.std"),
        "front_mean": t("observation.images.front.mean").view(1, 3, 1, 1),
        "front_std": t("observation.images.front.std").view(1, 3, 1, 1),
        "wrist_mean": t("observation.images.wrist.mean").view(1, 3, 1, 1),
        "wrist_std": t("observation.images.wrist.std").view(1, 3, 1, 1),
        "action_mean": torch.as_tensor(post["action.mean"]).to(device),
        "action_std": torch.as_tensor(post["action.std"]).to(device),
        # 数据集动作范围（安全限幅用，留在 CPU 上做 numpy clip）
        "action_min": post["action.min"].cpu().numpy(),
        "action_max": post["action.max"].cpu().numpy(),
    }

    print(
        f"策略加载完成：chunk_size={config.chunk_size}，"
        f"每 {config.n_action_steps} 步重规划一次"
    )
    return policy, norm


# ---------------- 相机 ----------------

class CameraThread(threading.Thread):
    """后台连续读帧，只保留最新一帧。

    打开参数与重试逻辑同 collect_data.py（DSHOW/MJPG/640x480/30fps）；
    区别是读取放线程里跑满帧率，避免推理端 10Hz 取帧时驱动缓冲积压
    旧帧。
    """

    def __init__(self, name, index):
        super().__init__(daemon=True, name=f"cam-{name}")
        self.name_key = name
        self.index = index
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0
        self.error = None

    def run(self):
        camera = None
        last_error = None

        for attempt in range(3):
            camera = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)

            if not camera.isOpened():
                last_error = (
                    f"无法打开相机 {self.name_key}，编号 {self.index}"
                )
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

            image = None
            for _ in range(10):
                ok, image = camera.read()
                if ok and image is not None:
                    break
                image = None

            if image is None:
                last_error = (
                    f"{self.name_key} 相机预热失败（第 {attempt+1} 次）"
                )
            elif image.shape != (HEIGHT, WIDTH, 3):
                last_error = (
                    f"{self.name_key} 实际分辨率 {image.shape}，"
                    f"与配置 {(HEIGHT, WIDTH, 3)} 不一致"
                )
            else:
                last_error = None
                break

            camera.release()
            camera = None
            time.sleep(1)

        if camera is None:
            self.error = last_error or "相机打开失败"
            return

        try:
            while not self.stop_event.is_set():
                ok, image = camera.read()

                if not ok or image is None:
                    # 单帧失败先容忍，连续失败靠取帧端超时发现
                    time.sleep(0.01)
                    continue

                if image.shape != (HEIGHT, WIDTH, 3):
                    self.error = f"{self.name_key} 相机分辨率发生变化"
                    break

                with self._lock:
                    self._frame = image
                    self._stamp = time.perf_counter()
        finally:
            camera.release()

    def latest(self, timeout=2.0):
        """返回 (frame_bgr, stamp)；超时或线程出错抛异常。"""
        deadline = time.perf_counter() + timeout

        while time.perf_counter() < deadline:
            if self.error:
                raise RuntimeError(self.error)

            with self._lock:
                if self._frame is not None:
                    return self._frame, self._stamp

            if not self.is_alive():
                raise RuntimeError(self.error or "相机线程退出")

            time.sleep(0.01)

        raise RuntimeError(f"{self.name_key} 相机 {timeout:.0f} 秒内无帧")

    def stop(self):
        self.stop_event.set()


# ---------------- 机器人 ----------------

def read_status(robot):
    """读全量状态并做完整性校验（同 collect_data.py：只有这个接口
    长时间稳定，轻量接口会被控制器返回错误码）。"""
    ret = robot.get_robot_status()

    if ret[0] != 0:
        raise RuntimeError(f"状态读取失败：{ret}")

    status = ret[1]

    if len(status) <= 22 or not status[22]:
        raise RuntimeError("状态格式异常或 SDK 通信断开")

    return status


def read_joints(robot):
    """从全量状态取六关节角（rad），status[19]。"""
    joints = np.asarray(read_status(robot)[19], dtype=float)

    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise RuntimeError("六轴关节数据异常")

    return joints


# 全量状态里各标志位的下标（与 1.7.2 文档一致，并已在真机上核对）：
# [2]上电 [3]使能 [6]拖拽模式 [22]SDK 通道 [23]急停
IDX_POWERED = 2
IDX_ENABLED = 3
IDX_DRAGGING = 6
IDX_ESTOP = 23


def robot_setup():
    """登录并确保机器人处于可运动状态（上电+使能+非拖拽）。

    登录调用方式同 collect_data.py。上电/使能按实际状态补做：
    已上电时再调 power_on 会被控制器拒绝并返回 -3（真机已验证），
    所以必须先查状态再决定调不调。is_in_drag_mode 等轻量接口在这台
    固件上不可用（返回 2），状态一律以全量接口为准。
    """
    sys.path.insert(0, SDK_DIR)
    os.add_dll_directory(SDK_DIR)

    import jkrc

    robot = jkrc.RC(ROBOT_IP)
    ret = robot.login()

    # 控制器冷启动时 login 常先返回 -1，需要重试（同 collect_data.py）
    attempts = 0
    while ret[0] != 0 and attempts < 10:
        time.sleep(2)
        ret = robot.login()
        attempts += 1

    if ret[0] != 0:
        raise RuntimeError(f"机器人登录失败：{ret}")

    print("机器人登录成功")

    status = read_status(robot)

    if status[IDX_ESTOP]:
        raise RuntimeError(
            "急停被按下，请先在示教器上解除急停再运行推理"
        )

    if status[IDX_DRAGGING]:
        print("检测到拖拽模式，正在退出……")
        robot.drag_mode_enable(False)
        time.sleep(0.5)
        status = read_status(robot)

        if status[IDX_DRAGGING]:
            raise RuntimeError(
                "无法退出拖拽模式，请在示教器上手动退出后重试"
            )

    if not status[IDX_POWERED]:
        print("机器人未上电，上电中（约需数秒）……")

        attempts = 0
        ret = robot.power_on()
        while ret[0] != 0 and attempts < 3:
            time.sleep(2)
            ret = robot.power_on()
            attempts += 1

        status = read_status(robot)

        if not status[IDX_POWERED]:
            raise RuntimeError(f"机器人上电失败：{ret}")
    else:
        print("机器人已上电")

    if not status[IDX_ENABLED]:
        print("机器人未使能，使能中……")
        time.sleep(0.5)

        attempts = 0
        ret = robot.enable_robot()
        while ret[0] != 0 and attempts < 3:
            time.sleep(2)
            ret = robot.enable_robot()
            attempts += 1

        status = read_status(robot)

        if not status[IDX_ENABLED]:
            raise RuntimeError(f"机器人使能失败：{ret}")
    else:
        print("机器人已使能，跳过上电/使能")

    # 消化首次调用约 190ms 的冷启动，让正式循环里的读取就是正常耗时
    for _ in range(3):
        robot.get_robot_status()
        time.sleep(0.05)

    return robot


# ---------------- 推理循环 ----------------

def to_tensor(image_bgr, mean, std, device):
    """BGR uint8 帧 → 归一化后的 (1,3,H,W) float 张量。"""
    import torch

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
    x = x.float().div_(255.0).unsqueeze(0)
    return (x - mean) / (std + NORM_EPS)


def observe(front, wrist, robot, norm, device):
    """取当前观测（最新两帧 + 关节角）并归一化成策略输入。

    robot 为 None（dry-run）时关节用 0 代替。返回 (batch, q, 图像龄)。
    """
    import torch

    frame_front, stamp_front = front.latest()
    frame_wrist, stamp_wrist = wrist.latest()

    q = read_joints(robot) if robot is not None else np.zeros(6)

    state = torch.from_numpy(q).float().to(device).unsqueeze(0)
    state = (state - norm["state_mean"]) / (
        norm["state_std"] + NORM_EPS
    )

    batch = {
        "observation.state": state,
        "observation.images.front": to_tensor(
            frame_front, norm["front_mean"], norm["front_std"], device
        ),
        "observation.images.wrist": to_tensor(
            frame_wrist, norm["wrist_mean"], norm["wrist_std"], device
        ),
    }

    now = time.perf_counter()
    age = max(now - stamp_front, now - stamp_wrist)

    return batch, q, age


def unnormalize_action(action_tensor, norm):
    """策略输出的归一化动作 → 目标关节角（rad，numpy）。"""
    return (
        action_tensor.squeeze(0).cpu().numpy()
        * (norm["action_std"].cpu().numpy() + NORM_EPS)
        + norm["action_mean"].cpu().numpy()
    )


def run(args):
    import torch

    device = args.device

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，退回 CPU（推理会很慢）")
        device = "cpu"

    policy, norm = load_policy(args.ckpt, device, args.replan_every)

    # 预热推理：首次 CUDA 调用约 1 秒，不能占控制周期的预算
    print("预热模型……")
    dummy = {
        "observation.state": torch.zeros(1, 6, device=device),
        "observation.images.front": torch.zeros(
            1, 3, HEIGHT, WIDTH, device=device
        ),
        "observation.images.wrist": torch.zeros(
            1, 3, HEIGHT, WIDTH, device=device
        ),
    }
    with torch.no_grad():
        policy.select_action(dummy)
    policy.reset()
    print("模型就绪")

    front = CameraThread("front", args.front)
    wrist = CameraThread("wrist", args.wrist)
    front.start()
    wrist.start()

    robot = None
    start_q = None
    log_file = None
    log_path = None

    try:
        if not args.dry_run:
            robot = robot_setup()
            q = read_joints(robot)
            print("\n当前关节角（rad）：")
            print(np.array2string(q, precision=3, floatmode="fixed"))
            print(
                "训练数据动作范围（rad）：\n  min "
                + np.array2string(
                    norm["action_min"], precision=3, floatmode="fixed"
                )
                + "\n  max "
                + np.array2string(
                    norm["action_max"], precision=3, floatmode="fixed"
                )
            )

            if np.any(q < norm["action_min"] - 0.05) or np.any(
                q > norm["action_max"] + 0.05
            ):
                print(
                    "警告：当前关节角超出训练数据范围，"
                    "策略在这个位姿下的输出不可信！"
                )
        else:
            print("dry-run：不连接机器人、不发送运动指令")

        # 相机线程内部有打开+预热+最多 3 次重试，给足首次等待时间
        front.latest(timeout=15)
        wrist.latest(timeout=15)
        print("相机就绪")

        print(
            "\n请把机器人和物体摆到示范采集时的初始状态，"
            "然后按 Enter 开始（Ctrl+C 退出）。",
            flush=True,
        )
        drain_keys()

        if args.yes:
            print("--yes：跳过确认直接开始", flush=True)
        else:
            while not enter_pressed():
                if robot is not None:
                    # 周期读状态保持 SDK 活跃（长时间闲置后首包会变慢）
                    read_joints(robot)
                time.sleep(0.5)

        # 起始关节：结束后回到这里，与示范的起点对齐
        start_q = (
            read_joints(robot) if robot is not None else np.zeros(6)
        )

        # 预览首步动作，增量异常直接拒绝执行
        batch, q_now, _ = observe(front, wrist, robot, norm, device)

        with torch.no_grad():
            action = policy.select_action(batch)

        target = unnormalize_action(action, norm)

        print("当前关节：", np.array2string(q_now, precision=3))
        print("首步目标：", np.array2string(target, precision=3))
        print("步进增量：", np.array2string(target - q_now, precision=3))

        if np.any(np.abs(target - q_now) > args.max_step):
            print(
                "首步目标距当前关节过远，可能初始位姿/场景与训练不符，"
                "已拒绝执行。检查初始状态，或必要时调大 --max-step。"
            )
            return

        # 预览产生的动作队列作废：倒计时 3 秒后循环必须从新鲜观测
        # 重新推理，否则前几步执行的是基于旧观测的动作
        policy.reset()

        print("3 秒后开始执行（Ctrl+C 立即中止）……", flush=True)
        for i in range(3, 0, -1):
            print(f"  {i}……", flush=True)
            time.sleep(1 if not args.yes else 0.2)

        # 每条 rollout 落一个 jsonl，便于复盘
        SAVE_ROOT.mkdir(parents=True, exist_ok=True)
        log_path = (
            SAVE_ROOT
            / f"rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        log_file = log_path.open("x", encoding="utf-8")

        period = 1.0 / CONTROL_FPS
        next_tick = time.perf_counter() + period
        last_print = time.perf_counter()

        print(
            f"\n开始执行 {args.steps} 步"
            f"（约 {args.steps / CONTROL_FPS:.0f} 秒），"
            "Enter 可提前停止。",
            flush=True,
        )

        for step in range(args.steps):
            tick_start = time.perf_counter()

            batch, q_now, img_age = observe(
                front, wrist, robot, norm, device
            )

            infer_start = time.perf_counter()

            with torch.no_grad():
                action = policy.select_action(batch)

            infer_ms = (time.perf_counter() - infer_start) * 1000

            target = unnormalize_action(action, norm)

            # 安全限幅 1：不超出训练数据的动作范围
            clamped = np.clip(
                target, norm["action_min"], norm["action_max"]
            )

            # 安全限幅 2：单步关节变化量上限，防异常输出导致猛跳
            delta = clamped - q_now
            max_delta = np.max(np.abs(delta))

            if max_delta > args.max_step:
                clamped = q_now + delta * (args.max_step / max_delta)

            move_ms = 0.0

            if robot is not None:
                move_start = time.perf_counter()
                ret = robot.joint_move(
                    clamped.tolist(), 0, False, args.speed
                )
                move_ms = (time.perf_counter() - move_start) * 1000

                if ret[0] != 0:
                    raise RuntimeError(
                        f"第 {step} 步 joint_move 失败：{ret}"
                    )

            log_file.write(json.dumps({
                "step": step,
                "t": tick_start,
                "q": q_now.tolist(),
                "target_raw": target.tolist(),
                "target_sent": clamped.tolist(),
                "img_age_s": img_age,
                "infer_ms": infer_ms,
                "move_ms": move_ms,
                "tick_ms": (time.perf_counter() - tick_start) * 1000,
            }) + "\n")
            log_file.flush()

            now = time.perf_counter()
            tick_ms = (now - tick_start) * 1000

            if now - last_print >= 1.0:
                print(
                    f"step {step + 1}/{args.steps}  "
                    f"tick {tick_ms:.0f}ms  infer {infer_ms:.0f}ms  "
                    f"Δmax {np.max(np.abs(clamped - q_now)):.3f}rad",
                    flush=True,
                )
                last_print = now

            if enter_pressed():
                print("\n收到 Enter，提前停止。", flush=True)
                break

            next_tick += period
            sleep_s = next_tick - time.perf_counter()

            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # 本步超时（重规划步可能到 140ms+），重置基准保住
                # 平均 10Hz，不让误差滚雪球
                next_tick = time.perf_counter()

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，中止。")

    finally:
        if robot is not None:
            try:
                robot.motion_abort()
                time.sleep(0.2)

                if (
                    not args.no_return
                    and start_q is not None
                ):
                    print("回到起始关节位置……")
                    ret = robot.joint_move(
                        start_q.tolist(), 0, True, args.speed
                    )

                    if ret[0] != 0:
                        robot.clear_error()
                        time.sleep(0.5)
                        ret = robot.joint_move(
                            start_q.tolist(), 0, True, args.speed
                        )

                    if ret[0] != 0:
                        print(f"回起始位置失败：{ret}")

                robot.logout()
                print("机器人已注销")
            except Exception as exc:  # noqa: BLE001
                print("收尾异常：", repr(exc))

        front.stop()
        wrist.stop()
        front.join(timeout=2)
        wrist.join(timeout=2)

        if log_file is not None:
            log_file.close()
            print("日志：", log_path)

    print("结束。")


def main():
    global ROBOT_IP

    parser = argparse.ArgumentParser(
        description="ACT 模型真机推理（JAKA ZU3 + lerobot）"
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_CKPT,
        help="pretrained_model 目录",
    )
    parser.add_argument("--ip", default=ROBOT_IP, help="机器人 IP")
    parser.add_argument(
        "--steps", type=int, default=100,
        help="执行步数（10 步/秒），默认 100 步 = 10 秒",
    )
    parser.add_argument(
        "--replan-every", type=int, default=10,
        help="每执行多少步用新观测重推一次动作块；0 表示用 checkpoint "
             "的 n_action_steps（=100，10 秒才重规划一次）",
    )
    parser.add_argument(
        "--speed", type=float, default=0.8,
        help="joint_move 速度（rad/s）",
    )
    parser.add_argument(
        "--max-step", type=float, default=0.15,
        help="单步（0.1 秒）关节角最大变化量（rad），超过则等比收缩",
    )
    parser.add_argument(
        "--front", type=int, default=CAMERA_FRONT_DEFAULT,
        help="front 相机编号",
    )
    parser.add_argument(
        "--wrist", type=int, default=CAMERA_WRIST_DEFAULT,
        help="wrist 相机编号",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只跑相机+推理，不连接机器人、不发运动指令",
    )
    parser.add_argument(
        "--no-return", action="store_true",
        help="结束后不回到起始关节位置",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="跳过按 Enter 确认和倒数（用于自动化测试）",
    )

    args = parser.parse_args()

    if args.front == args.wrist:
        parser.error("两个相机编号不能相同")

    if args.steps < 1:
        parser.error("--steps 至少为 1")

    ROBOT_IP = args.ip

    run(args)


if __name__ == "__main__":
    main()
