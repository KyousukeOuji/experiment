import numpy as np
import k4a


class Kinect:
    # kinect device
    device = None
    capture = None

    # color
    color_data = None
    color_arr = None
    # depth
    depth_data = None
    depth_arr = None
    # resolution
    width = 1280
    height = 720
    center = (width/2, height/2)

    def __init__(self):
        self.initialize()

    def __del__(self):
        self.finalize()

    def initialize(self):
        self.initialize_sensor()

    def finalize(self):
        self.finalize_sensor()

    def initialize_sensor(self, device_index=0):
        # open device
        self.device = k4a.Device.open(device_index)

        # start cameras
        device_config = k4a.DeviceConfiguration(
                color_format=k4a.EImageFormat.COLOR_BGRA32,
                color_resolution=k4a.EColorResolution.RES_720P,
                depth_mode=k4a.EDepthMode.NFOV_UNBINNED,
                camera_fps=k4a.EFramesPerSecond.FPS_15,
                synchronized_images_only=True,
                )
        status = self.device.start_cameras(device_config)
        if status != k4a.EStatus.SUCCEEDED:
            raise IOError("failed starting cameras!")
        # k4a.DeviceSetColorControl(self.device, k4a.ColorControlCommand.BRIGHTNESS, k4a.ColorControlMode.MANUAL, 50)

    def finalize_sensor(self):
        # stop cameras
        self.device.stop_cameras()
        self.device.close()

    def update(self):
        self.update_frame()
        self.update_color()
        self.update_depth()

    def update_frame(self):
        # capture frame
        self.capture = self.device.get_capture(-1)   #デバイスから最新のフレームを取得
        if self.capture is None:
            raise IOError("failed getting capture!")

    def update_color(self):
        self.color_data = self.capture.color.data
        self.color_arr = np.asarray(self.color_data)

    def update_depth(self):
        self.depth_data = self.capture.depth.data
        self.depth_arr = np.asarray(self.depth_data)
