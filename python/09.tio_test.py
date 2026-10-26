import time
import jkrc
'''
Take the DH Gripper as an example
'''

demo = jkrc.RC("10.5.5.100")
demo.login()
demo.power_on()
# must disable robot before setting tio params
demo.disable_robot()
print(demo.get_sdk_version())

chnid = 2
chnmode = 0

mod_rtu_comm = {
    'chn_id':1,
    'slave_id':1,
    'baudrate': 115200,
    'databit': 8,
    'stopbit': 1,
    'parity': 78
}
# set params
demo.set_tio_vout_param(1,0)  # ebable tio； 24V
demo.set_tio_pin_mode(2,1)    # rs485l; Ai
time.sleep(1)
demo.set_rs485_chn_mode(1,0)
r = demo.set_rs485_chn_comm({
    'chn_id':1,
    'slave_id':1,
    'baudrate': 115200,
    'databit': 8,
    'stopbit': 1,
    'parity': 78
})

demo.enable_robot()

# send command 
# init
command = bytearray.fromhex("01 06 01 00 00 01")  # must be bytearray object
ret = demo.send_tio_rs_command(2, command)
print("==", ret)
time.sleep(2)
# open 500
command2 = bytearray.fromhex("01 06 01 03 01 F4")
ret1 = demo.send_tio_rs_command(2, command2)
print(ret1)
ret2 = demo.get_rs485_chn_comm()
print(ret2)
