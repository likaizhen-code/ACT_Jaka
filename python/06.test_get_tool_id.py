import math
import time
import __common

def main():
    ###################### 测试代码开始 ######################
    rc = jkrc.RC("192.168.137.173")
    print('login 192.168.137.173: {}'.format(rc.login()))
    print('power_on: {}'.format(rc.power_on()))
    print('enable_robot: {}'.format(rc.enable_robot()))

    while True:
        print("===================")

        id = rc.get_tool_id()

        print('robot id: {}'.format(id))

    ###################### 测试代码结束 ######################

    print('disable_robot: {}'.format(rc.disable_robot()))
    print('power_off: {}'.format(rc.power_off()))
    print('logout: {}'.format(rc.logout()))


if __name__ == '__main__':
    __common.init_env()
    import jkrc

    main()