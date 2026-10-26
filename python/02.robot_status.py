import os
import sys
import time

# 加载 SDK
sdk = r"E:\JAKA_data_collect\Windows\python3\x64"
sys.path.insert(0, sdk)
dll_handle = os.add_dll_directory(sdk)

import jkrc


def main():
    robot = jkrc.RC("10.5.5.100")
    ret = robot.login()

    if ret[0] != 0:
        raise RuntimeError(f"登录失败：{ret}")

    print("登录成功，按 Ctrl+C 停止。")

    try:
        while True:
            ret = robot.get_robot_status()

            if ret[0] != 0:
                raise RuntimeError(f"读取状态失败：{ret}")

            print(ret, flush=True)
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n已停止。")

    finally:
        robot.logout()


if __name__ == "__main__":
    main()