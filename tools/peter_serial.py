import serial as ser
import time
import math
import numpy as np

class SerialAction:
    def __init__(self, serial_config_dict):
        '''
        //最多15个值  
        //字符格式发送 无换行无回车 
        
        //****通讯案例****//
        //M为运动模式  C为运动控制  D为数据控制
        //MAM30B15W15R15E	运动模式A CPG参数设置 
        //——————单侧摆幅30°   			《最大50°》
        //——————中位偏转15°  				《最大30°》	
        //——————摆动频率1.5hz				《最大4hz》
        //——————转弯周期内的时间比1.5：1  《最小1：1 最大4：1》
            
        //CFE	直游
        //CLE	左转
        //CRE 右转
        //CSE 停止
        
        //DA1E 上位机开启角度数据接收 // DA0E 上位机关闭角度数据接收 // DA2E 上位机开启角度数据标零
        //DI1E 上位机开启电流数据接收 // DI0E 上位机关闭电流数据接收
        ''' 
        self.port = serial_config_dict["port"]
        self.baudrate = serial_config_dict["baudrate"]
        self.duration = serial_config_dict["duration"]
        self.fish_serial = ser.Serial(
            port=self.port,
            baudrate=self.baudrate,
            parity=ser.PARITY_NONE,
            stopbits=ser.STOPBITS_ONE,
            bytesize=ser.EIGHTBITS,
            timeout=None,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False
        )
        print("Serial port is opened!")
        self.serial_state()
        # self.action_list = None
        self.motion_state = None
        self.amp = None
        self.offset = None
        self.omega = None
        self.ratio = None

        self.control_type = serial_config_dict["control_type"]
        if self.control_type == "manual":
            if serial_config_dict["action_list"] is None:
                raise ValueError("action_list is None")
            self.action_list = serial_config_dict["action_list"]

    def serial_state(self):
        # 打印串口配置
        print(f"设备状态 {self.port}:")
        print("----------")
        print(f"    波特率:     {self.fish_serial.baudrate}")
        print(f"    奇偶校验:   {self.fish_serial.parity}")
        print(f"    数据位:     {self.fish_serial.bytesize}")
        print(f"    停止位:     {self.fish_serial.stopbits}")
        print(f"    超时:       {'OFF' if self.fish_serial.timeout is None else self.fish_serial.timeout}")
        print(f"    XON/XOFF:   {'ON' if self.fish_serial.xonxoff else 'OFF'}")
        print(f"    CTS 握手:   {'ON' if self.fish_serial.rtscts else 'OFF'}")
        print(f"    DSR 握手:   {'ON' if self.fish_serial.dsrdtr else 'OFF'}")
        print(f"    DSR 敏感度: {'ON' if self.fish_serial.dsrdtr else 'OFF'}")
        print(f"    DTR 电路:   {'ON' if self.fish_serial.dtr else 'OFF'}")
        print(f"    RTS 电路:   {'ON' if self.fish_serial.rts else 'OFF'}")

    def fix_route(self):
        '''
        按照预设动作执行
        '''
        for i in range(len(self.action_list)):
            self.send(self.action_list[i][0],self.action_list[i][1],self.action_list[i][2],self.action_list[i][3],self.action_list[i][4])
            if self.action_list[i][5] is not None:
                time.sleep(self.action_list[i][5])
            else:
                time.sleep(self.duration)
        self.send("CSE",None,None,None,None)
        print("Fix route is over!")
         
    def keyboard_control(self):
        '''
        args:
            w: 前进, CFE
            s: 停止, CSE
            a: 左转, CLE
            d: 右转, CRE
            l: 下一组动作
            j: 上一组动作
            f: 按照预设动作执行
            q: 退出
        '''
        #默认使用第一组控制参数
        ptr = 0

        self.fish_serial.write(f"MAM{self.action_list[ptr]['amp']}B{self.action_list[ptr]['offset']}W{self.action_list[ptr]['omega']}R{self.action_list[ptr]['ratio']}E".encode())
        time.sleep(1)
        while True:
            key = input("Please input the control key:")
            if key == "w":
                self.fish_serial.write("CFE".encode())
            elif key == "s":
                self.fish_serial.write("CSE".encode())
            elif key == "a":
                self.fish_serial.write("CLE".encode())
            elif key == "d":
                self.fish_serial.write("CRE".encode())
            elif key == "l":
                ptr = ptr+1
                if ptr >= (len(self.action_list)-1):
                    ptr = len(self.action_list)-1
                    print("This is the end of the action list!")
                self.send(self.action_list[ptr]['motion_state'],self.action_list[ptr]['amp'],self.action_list[ptr]['offset'],self.action_list[ptr]['omega'],self.action_list[ptr]['ratio'])
                print(ptr)
                    
            elif key == "j":
                ptr = ptr-1
                if ptr <= 0:
                    ptr = 0
                    print("This is the beginning of the action list!")
                self.send(self.action_list[ptr]['motion_state'],self.action_list[ptr]['amp'],self.action_list[ptr]['offset'],self.action_list[ptr]['omega'],self.action_list[ptr]['ratio'])
                print(ptr)

            elif key == "f":
                self.fix_route()
            elif key == "q":
                break
        print("keyboard control is over!")
        self.fish_serial.close()
        print("Serial port is closed!")
    

    def send(self,my_motion_state,my_amp = None,my_offset = None,my_omega = None,my_ratio = None):
        self.motion_state = my_motion_state
        print(self.motion_state)
        if self.motion_state == "CSE":
             self.fish_serial.write(self.motion_state.encode())
        else:
            self.amp = my_amp
            self.offset = my_offset
            self.omega = my_omega
            self.ratio = my_ratio
            self.fish_serial.write(f"MAM{self.amp}B{self.offset}W{self.omega}R{self.ratio}E".encode())
            print(f"MAM{self.amp}B{self.offset}W{self.omega}R{self.ratio}E")
            time.sleep(1)   
            self.fish_serial.write(self.motion_state.encode())

    def close(self):
        self.fish_serial.close()
        print("Serial port is closed!")


def main():

    my_action_dict_1 = {
        "motion_state": "CFE",
        "amp": '00',
        "offset": '15',
        "omega": '10',
        "ratio": '10',
        "duration": 5,
    }
    my_action_dict_2 = {
        "motion_state": "CLE",
        "amp": '35',
        "offset": '15',
        "omega": '20',
        "ratio": '10',
        "duration": 5,
    }
    my_action_dict_3 = {
        "motion_state": "CRE",
        "amp": '40',
        "offset": '15',
        "omega": '15',
        "ratio": '10',
        "duration": 5,
    }
    my_action_dict_4 = {
        "motion_state": "CSE",
        "amp": '40',
        "offset": '15',
        "omega": '20',
        "ratio": '10',
        "duration": 5,
    }
    
    my_serial_config_dict = {
        "port": "COM7",
        "baudrate": 115200,
        "duration": 5,
        "control_type": "manual",
        "action_list": [my_action_dict_1, my_action_dict_2, my_action_dict_3, my_action_dict_4],
    }

    my_serial_action = SerialAction(my_serial_config_dict)
    my_serial_action.keyboard_control()

if __name__ == "__main__":
    main()  