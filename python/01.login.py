
import __common

def main():
    rc = jkrc.RC("192.168.137.164")
    rc.login()
    rc.logout()

if __name__ == '__main__':
    __common.init_env()
    import jkrc

    main()