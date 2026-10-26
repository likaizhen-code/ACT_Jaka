# coding=utf-8
import __common
import math
def d2r(v):
    return v / 180.0 * math.pi

def main():
    rc = jkrc.RC("192.168.137.173")
    rc.login()
    rc.power_on()
    rc.enable_robot()
    res = rc.linear_move([-307.540, 116.21, 291.55, d2r(180), d2r(0), d2r(90)], 0, 1, 10)
    print('linear move res is {}'.format(res))
    # rc.clear_error()
    robot_status = rc.get_robot_status()
    # print('get robot status {}'.format(res))
    print(res)
    rc.logout()

if __name__ == '__main__':
    __common.init_env()
    import jkrc

    main()