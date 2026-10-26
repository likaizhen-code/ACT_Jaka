# coding=utf-8

import __common
import sys
import platform

__system = platform.system()

def assert_0_or_exit(ret, msg):
    if not ret == 0:
        print('{} ({})failed and exit.'.format(msg, ret))
        sys.exit(-1)
    else:
        print('{} succ.'.format(msg))

def main():
    rc = jkrc.RC('192.168.137.173')
    rc.login()

    if __system == 'Windows':
        local = r'C:\Users\jenkins\Desktop\sdk_dev\jakaAPI-c\jakaAPI-c\testfile'
        local_download = r'C:\Users\jenkins\Desktop\sdk_dev\jakaAPI-c\jakaAPI-c\testfile.tmp'
    else:
        local = r'/home/zgj/sdk_dev/jakaAPI-c/testfile'
        local_download = r'/home/zgj/sdk_dev/jakaAPI-c/testfile.tmp'
    remote = 'log/testfile'
    remote_new = 'log/testfile.tmp'

    # 文件操作
    ret = rc.upload_file(local, remote, 1)
    assert_0_or_exit(ret[0], 'upload file')

    ret = rc.rename_ftp_file(remote, remote_new, 1)
    assert_0_or_exit(ret[0], 'rename file')

    ret = rc.download_file(local_download, remote_new, 1)
    assert_0_or_exit(ret[0], 'download file')

    ret = rc.del_ftp_file(remote_new, 1)
    assert_0_or_exit(ret[0], 'del file')

    if __system == 'Windows':
        local = r'C:\Users\jenkins\Desktop\sdk_dev\jakaAPI-c\jakaAPI-c\test_dir'
        local_download = r'C:\Users\jenkins\Desktop\sdk_dev\jakaAPI-c\jakaAPI-c\test_dir.tmp'
    else:
        local = r'/home/zgj/sdk_dev/jakaAPI-c/test_dir'
        local_download = r'/home/zgj/sdk_dev/jakaAPI-c/test_dir.tmp'
    remote = 'log/test_dir'
    remote_new = 'log/test_dir.tmp'

    # 目录操作
    ret = rc.upload_file(local, remote, 2)
    assert_0_or_exit(ret[0], 'upload dir')

    ret = rc.rename_ftp_file(remote, remote_new, 2)
    assert_0_or_exit(ret[0], 'rename dir')

    ret = rc.download_file(local_download, remote_new, 2)
    assert_0_or_exit(ret[0], 'download dir')

    ret = rc.del_ftp_file(remote_new, 2)
    assert_0_or_exit(ret[0], 'del dir')

    if __system == 'Windows':
        local = r'C:\Users\jenkins\Desktop\sdk_dev\jakaAPI-c\jakaAPI-c\测试文件'
        local_download = r'C:\Users\jenkins\Desktop\sdk_dev\jakaAPI-c\jakaAPI-c\测试文件.tmp'
    else:
        local = r'/home/zgj/sdk_dev/jakaAPI-c/测试文件'
        local_download = r'/home/zgj/sdk_dev/jakaAPI-c/测试文件.tmp'
    remote = 'log/测试文件'
    remote_new = 'log/测试文件.tmp'

    # 文件操作
    ret = rc.upload_file(local, remote, 1)
    assert_0_or_exit(ret[0], 'upload file')

    ret = rc.rename_ftp_file(remote, remote_new, 1)
    assert_0_or_exit(ret[0], 'rename file')

    ret = rc.download_file(local_download, remote_new, 1)
    assert_0_or_exit(ret[0], 'download file')

    ret = rc.del_ftp_file(remote_new, 1)
    assert_0_or_exit(ret[0], 'del file')

    if __system == 'Windows':
        local = r'C:\Users\jenkins\Desktop\sdk_dev\jakaAPI-c\jakaAPI-c\测试目录'
        local_download = r'C:\Users\jenkins\Desktop\sdk_dev\jakaAPI-c\jakaAPI-c\测试目录.tmp'
    else:
        local = r'/home/zgj/sdk_dev/jakaAPI-c/测试目录'
        local_download = r'/home/zgj/sdk_dev/jakaAPI-c/测试目录.tmp'
    remote = 'log/测试目录'
    remote_new = 'log/测试目录.tmp'

    # 目录操作
    ret = rc.upload_file(local, remote, 2)
    assert_0_or_exit(ret[0], 'upload dir')

    ret = rc.rename_ftp_file(remote, remote_new, 2)
    assert_0_or_exit(ret[0], 'rename dir')

    ret = rc.download_file(local_download, remote_new, 2)
    assert_0_or_exit(ret[0], 'download dir')

    ret = rc.del_ftp_file(remote_new, 2)
    assert_0_or_exit(ret[0], 'del dir')

    if __system == 'Windows':
        local = r'C:\Users\jenkins\Desktop\sdk_dev\jakaAPI-c\jakaAPI-c\测试 文件'
        local_download = r'C:\Users\jenkins\Desktop\sdk_dev\jakaAPI-c\jakaAPI-c\测试 文件.tmp'
    else:
        local = r'/home/zgj/sdk_dev/jakaAPI-c/测试 文件'
        local_download = r'/home/zgj/sdk_dev/jakaAPI-c/测试 文件.tmp'
    remote = 'log/测试 文件'
    remote_new = 'log/测试 文件.tmp'

    # 文件操作
    ret = rc.upload_file(local, remote, 1)
    assert_0_or_exit(ret[0], 'upload file')

    ret = rc.rename_ftp_file(remote, remote_new, 1)
    assert_0_or_exit(ret[0], 'rename file')

    ret = rc.download_file(local_download, remote_new, 1)
    assert_0_or_exit(ret[0], 'download file')

    ret = rc.del_ftp_file(remote_new, 1)
    assert_0_or_exit(ret[0], 'del file')

    if __system == 'Windows':
        local = r'C:\Users\jenkins\Desktop\sdk_dev\jakaAPI-c\jakaAPI-c\测试 目录'
        local_download = r'C:\Users\jenkins\Desktop\sdk_dev\jakaAPI-c\jakaAPI-c\测试 目录.tmp'
    else:
        local = r'/home/zgj/sdk_dev/jakaAPI-c/测试 目录'
        local_download = r'/home/zgj/sdk_dev/jakaAPI-c/测试 目录.tmp'
    remote = 'log/测试 目录'
    remote_new = 'log/测试 目录.tmp'

    # 目录操作
    ret = rc.upload_file(local, remote, 2)
    assert_0_or_exit(ret[0], 'upload dir')

    ret = rc.rename_ftp_file(remote, remote_new, 2)
    assert_0_or_exit(ret[0], 'rename dir')

    ret = rc.download_file(local_download, remote_new, 2)
    assert_0_or_exit(ret[0], 'download dir')

    # 查看文件列表
    remote_get_dir = 'log'
    ret = rc.get_ftp_dir(remote_get_dir, 0)
    assert_0_or_exit(ret[0], 'remote get dir')
    print('get res: {}\n{}'.format(ret[0], ret[1]))

    ret = rc.del_ftp_file(remote_new, 2)
    assert_0_or_exit(ret[0], 'del dir')

    rc.logout()

if __name__ == '__main__':
    __common.init_env()
    import jkrc

    main()