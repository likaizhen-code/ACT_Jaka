# -*- coding: utf-8
import os
import sys
import ctypes

import platform

# 获取当前系统名称
__system = platform.system()

if __system == "Windows":
    print("当前系统为Windows")
    def init_env():
        base_dir = os.path.abspath('.')
        print('base dir', base_dir)

        # 指定 Windows 动态库查找路径
        syspath = os.path.join(base_dir, r'out\python3\Release')
        sys.path.append(syspath)

        # 指定 Windows Python jkzuc模块查找路径
        env_path = os.path.join(base_dir, r'out\shared\Release')
        path_env = os.environ.get('PATH')
        path_env = env_path + ';' + path_env
        os.environ['PATH'] = path_env
        # print('env: {}'.format(os.environ['PATH']))
elif __system == "Linux":
    print("当前系统为Linux")
    def init_env():
        base_dir = os.path.abspath('.')

        # 加载 Linux 动态库 libjakaAPI.so
        env_path = os.path.join(base_dir, r'out/shared/libjakaAPI.so')
        ctypes.CDLL(env_path)

        # 加载 Linux Python jkzuc模块查找路径
        syspath = os.path.join(base_dir, r'out/python3')
        sys.path.append(syspath)
        # print('SYS PATH: {}\n {}'.format(sys.path, syspath))
        # print('LD_LIBRARY_PATH: {}'.format(os.environ['LD_LIBRARY_PATH']))
else:
    print("未知系统")


if __name__ == '__main__':
    init_env()
