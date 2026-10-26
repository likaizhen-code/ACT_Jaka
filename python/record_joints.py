import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 沿用当前能正常读取完整状态的 SDK
sdk = r"E:\jaka\JAKA_data_collect\Windows\python3\x64"
sys.path.insert(0, sdk)
dll_handle = os.add_dll_directory(sdk)

import jkrc


def main():
    folder = Path(r"E:\jaka\SDK v2.2.2")
    folder.mkdir(parents=True, exist_ok=True)
    filename = folder / (
        f"joints_{datetime.now():%Y%m%d_%H%M%S_%f}.txt"
    )

    robot = jkrc.RC("10.5.5.100")
    ret = robot.login()
    if ret[0] != 0:
        raise RuntimeError(f"登录失败：{ret}")

    try:
        with filename.open("x", encoding="utf-8") as f:
            f.write(
                "time_s\tq1_rad\tq2_rad\tq3_rad\t"
                "q4_rad\tq5_rad\tq6_rad\n"
            )

            start = time.perf_counter()
            print("保存到：", filename)
            print("开始记录，按 Ctrl+C 停止。")

            while True:
                # 保留已能正常运行的完整状态接口
                ret = robot.get_robot_status()
                elapsed = time.perf_counter() - start

                if ret[0] != 0:
                    raise RuntimeError(f"读取状态失败：{ret}")

                status = ret[1]
                if len(status) <= 22:
                    raise RuntimeError("状态结构与当前版本不一致")

                if not status[22]:
                    raise RuntimeError("状态报告 SDK 通信连接异常")

                # 第 20 项：六轴关节角，单位为弧度
                joints = status[19]
                if not isinstance(joints, (tuple, list)) or len(joints) != 6:
                    raise RuntimeError(f"关节数据格式异常：{joints}")

                line = "\t".join(
                    [f"{elapsed:.6f}"]
                    + [f"{q:.9f}" for q in joints]
                )

                f.write(line + "\n")
                f.flush()
                print(line, flush=True)

                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n已停止记录。")

    finally:
        robot.logout()


if __name__ == "__main__":
    main()