from os import device_encoding
import k4a

from config.kinect_config import KinectConfig
# from util.custom_capture import CustomCapture

# できればTransformはCustomCaptureのフィールドとして持ちたい
# CustomCaptureでtransform済みのdepthを返せれば理想
class Kinect:
    def __init__(self, device):
        if not isinstance(device, k4a.Device):
            raise Exception("device must be an instance of class k4a.Device")
        self.__device = device
        self.__config = KinectConfig()
        self.__transform = None

    def setup(self):
        color_control_mode, color_control_commands = self._get_color_control_config()
        for color_control_command, value in color_control_commands:
            status = self.__device.set_color_control(
                color_control_command,
                color_control_mode,
                value
            )
            if status != k4a.EStatus.SUCCEEDED:
                raise Exception("failed to set color control")

        self.start_camera()

        # 露光時間の確認用
        (saved_value, mode) = self.__device.get_color_control(
            k4a.EColorControlCommand.EXPOSURE_TIME_ABSOLUTE
        )
        print(f"exposure time: {saved_value}")
        print(f"exposure mode: {mode}")

        calibration = self.get_calibration()
        self.__transform = k4a.Transformation(calibration)

        print("setup successfully finished")
    
    def start_camera(self):
        device_config = self._get_device_config()
        status = self.__device.start_cameras(device_config)
        if status != k4a.EStatus.SUCCEEDED:
            raise Exception("failed to start camera")

    def get_calibration(self):
        device_config = self._get_device_config()
        calibration = self.__device.get_calibration(
            depth_mode = device_config.depth_mode,
            color_resolution = device_config.color_resolution,
        )
        return calibration
    
    def _get_device_config(self):
        return self.__config.get_device_config()
    
    def _get_color_control_config(self):
        return self.__config.get_color_control_config()
        
    def capture(self):
        return self.__device.get_capture(-1)
    # def capture(self):
    #     capture = self.__device.get_capture(-1)
    #     calibration = self.get_calibration()
    #     custom_capture = CustomCapture(capture, calibration)
    #     return custom_capture

    def get_transform(self):
        if self.__transform == None:
            calibration = self.get_calibration()
            self.__transform = k4a.Transformation.create(calibration)
        return self.__transform
