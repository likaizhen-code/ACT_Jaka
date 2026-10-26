# -*- coding: utf-8 -*-
"""把机械臂移动到示范采集的初始关节位姿。

起始位姿取自训练数据集（dataset_jaka_zu3，59 条示范第一帧的均值，
各条之间极差仅 0.04~0.27 rad，属人工摆位抖动）：
    [2.970, 1.392, 1.476, 1.927, -1.637, -0.002] rad

用途：推理/采集开始前把机械臂复位到与示范一致的起点。

用法（本目录下，lerobot 或 jaka39 环境均可）：
  python init_pose.py            # 显示当前/目标位姿，按 Enter 确认后移动
  python init_pose.py --yes      # 跳过确认直接移动
  python init_pose.py --check    # 只显示当前/目标差异，不等待也不移动
  python init_pose.py --joints "3.0,1.4,1.48,1.9,-1.64,0"   # 自定义目标
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import msvcrt
except ImportError:
    msvcrt = None

# 与 collect_data.py 相同的 SDK 路径与机器人地址
SDK_DIR = r"E:\jaka\JAKA_data_collect\Windows\python3\x64"
ROBOT_IP = "10.5.5.100"

# 训练数据集 59 条示范第一帧关节角的均值（rad）
HOME_JOINTS = np.array(
    [2.9702, 1.3920, 1.4759, 1.9267, -1.6366, -0.0017],
    dtype=float,
)

# 到位判定容差（rad）
ARRIVE_TOL = 0.05

# 全量状态里各标志位的下标（与 act_infer.py 一致，真机核对过）：
# [2]上电 [3]使能 [6]拖拽模式 [22]SDK 通道 [23]急停
IDX_POWERED = 2
IDX_ENABLED = 3
IDX_DRAGGING = 6
IDX_ESTOP = 23


def enter_pressed():
    return (
        msvcrt is not None
        and msvcrt.kbhit()
        and msvcrt.getwch() in ("\r", "\n")
    )


def read_status(robot):
    """读全量状态并校验（同 collect_data.py：只有这个接口长时间稳定）。"""
    ret = robot.get_robot_status()

    if ret[0] != 0:
        raise RuntimeError(f"状态读取失败：{ret}")

    status = ret[1]

    if len(status) <= 22 or not status[22]:
        raise RuntimeError("状态格式异常或 SDK 通信断开")

    return status


def read_joints(robot):
    joints = np.asarray(read_status(robot)[19], dtype=float)

    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise RuntimeError("六轴关节数据异常")

    return joints


def robot_setup():
    """登录并确保可运动状态（上电+使能+非拖拽）。

    已上电时再调 power_on 会被拒绝返回 -3（真机验证过），所以按
    实际状态决定是否调用。
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
        raise RuntimeError("急停被按下，请先在示教器上解除急停")

    if status[IDX_DRAGGING]:
        print("检测到拖拽模式，正在退出……")
        robot.drag_mode_enable(False)
        time.sleep(0.5)
        status = read_status(robot)

        if status[IDX_DRAGGING]:
            raise RuntimeError("无法退出拖拽模式，请在示教器上手动退出")

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
        print("机器人已使能")

    return robot


def main():
    global ROBOT_IP

    parser = argparse.ArgumentParser(
        description="移动机械臂到示范初始关节位姿"
    )
    parser.add_argument("--ip", default=ROBOT_IP, help="机器人 IP")
    parser.add_argument(
        "--joints",
        help='自定义目标关节角，逗号分隔 6 个数（rad），'
             '如 "3.0,1.4,1.48,1.9,-1.64,0"',
    )
    parser.add_argument(
        "--speed", type=float, default=0.5, help="运动速度（rad/s）"
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="跳过按 Enter 确认"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="只显示当前/目标差异，不等待也不移动",
    )
    args = parser.parse_args()

    ROBOT_IP = args.ip

    target = HOME_JOINTS.copy()

    if args.joints:
        target = np.array(
            [float(v) for v in args.joints.split(",")], dtype=float
        )

        if target.shape != (6,):
            parser.error("--joints 需要 6 个逗号分隔的数值")

    robot = robot_setup()

    try:
        current = read_joints(robot)

        print("\n当前关节（rad）：", np.array2string(current, precision=3))
        print("目标关节（rad）：", np.array2string(target, precision=3))
        print(
            "差值（rad）：    ",
            np.array2string(target - current, precision=3),
        )
        print(
            f"最大关节差：{np.max(np.abs(target - current)):.3f} rad "
            f"({np.degrees(np.max(np.abs(target - current))):.1f}°)"
        )

        if args.check:
            print("\n--check：仅查看，不移动。")
            return

        if np.all(np.abs(target - current) < ARRIVE_TOL):
            print("\n已在目标位姿容差范围内，无需移动。")
            return

        if not args.yes:
            print(
                "\n确认路径无障碍物后按 Enter 开始移动（Ctrl+C 取消）……",
                flush=True,
            )
            while not enter_pressed():
                time.sleep(0.2)

        # 清掉可能残留的运动指令，再执行这次移动
        robot.motion_abort()
        time.sleep(0.1)

        print(f"以 {args.speed} rad/s 移动到目标位姿……", flush=True)
        ret = robot.joint_move(target.tolist(), 0, True, args.speed)

        if ret[0] != 0:
            raise RuntimeError(f"joint_move 失败：{ret}")

        final = read_joints(robot)
        err = np.max(np.abs(target - final))
        print("到位后关节（rad）：", np.array2string(final, precision=3))

        if err < ARRIVE_TOL:
            print(f"初始化完成，最大偏差 {err:.4f} rad。")
        else:
            print(
                f"警告：到位偏差 {err:.4f} rad 超过容差 {ARRIVE_TOL}，"
                "请检查是否有物体阻挡或碰撞保护。"
            )

    finally:
        robot.logout()
        print("机器人已注销")


if __name__ == "__main__":
    main()
