import k4a

class KinectConfig:
    def __init__(self):
        self.__device_config = k4a.DeviceConfiguration(
            color_format=k4a.EImageFormat.COLOR_BGRA32,
            color_resolution=k4a.EColorResolution.RES_720P,
            depth_mode=k4a.EDepthMode.NFOV_2X2BINNED,
            camera_fps=k4a.EFramesPerSecond.FPS_15,
            synchronized_images_only=True,
        )
        self.__color_control_mode = ColorControlMode(k4a.EColorControlMode.MANUAL)
        self.__color_control_commands = ColorControlCommands([
            # 露光時間を設定
            ColorControlCommand(k4a.EColorControlCommand.EXPOSURE_TIME_ABSOLUTE, 2500),
            #  ホワイトバランスを固定
            ColorControlCommand(k4a.EColorControlCommand.WHITEBALANCE, 4500),
            # 明るさを設定
            ColorControlCommand(k4a.EColorControlCommand.BRIGHTNESS, 128),
            # コントラストを設定
            ColorControlCommand(k4a.EColorControlCommand.CONTRAST, 5),
            # 彩度を設定
            ColorControlCommand(k4a.EColorControlCommand.SATURATION, 32),
            # シャープネスを設定
            ColorControlCommand(k4a.EColorControlCommand.SHARPNESS, 2),
            # ゲインを設定
            ColorControlCommand(k4a.EColorControlCommand.GAIN, 50),
            # バックライド補償？をOFF
            ColorControlCommand(k4a.EColorControlCommand.BACKLIGHT_COMPENSATION, 0),
            # 電源の周波数を60Hzに設定
            ColorControlCommand(k4a.EColorControlCommand.POWERLINE_FREQUENCY, 2)
        ])   
    def get_device_config(self):
        return self.__device_config
    def get_color_control_config(self):
        return self.__color_control_mode.get(), self.__color_control_commands.get()

class ColorControlMode:
    def __init__(self, color_control_mode):
        if not isinstance(color_control_mode, k4a.EColorControlMode):
            raise Exception("color control mode must be an instance of class k4a.EColorControlMode")
        self.__color_control_mode = color_control_mode
    def get(self):
        return self.__color_control_mode

class ColorControlCommands:
    def __init__(self, color_controls: list):
        self.__color_controls = []
        for color_control in color_controls:
            if not isinstance(color_control, ColorControlCommand):
                raise Exception("color control command must be an instance of class ColorControl")
            self.__color_controls.append(color_control)
    def get(self):
        color_controls_of_touple = []
        for color_control in self.__color_controls:
            color_controls_of_touple.append(color_control.get())
        return color_controls_of_touple

class ColorControlCommand:
    def __init__(self, color_control_command, value):
        if not isinstance(color_control_command, k4a.EColorControlCommand):
            raise Exception("color control command must be an instance of class k4a.EColorControlCommand")
        if not isinstance(value, int):
            raise Exception("color control command must be int")
        self.__color_control_command = color_control_command
        self.__value = value
    def get(self):
        return self.__color_control_command, self.__value
