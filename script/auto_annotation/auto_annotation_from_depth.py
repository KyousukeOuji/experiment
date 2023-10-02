# This package is used for displaying the images.
# It is not part of the k4a package and is not a hard requirement for k4a.
# Users need to install these packages in order to use this module.
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# This will import all the public symbols into the k4a namespace.
import k4a

def plot_images(image1:k4a.Image, image2:k4a.Image, image3, cmap:str=''):

    # Create figure and subplots.
    fig = plt.figure()
    ax = []
    ax.append(fig.add_subplot(1, 3, 1, label="Color"))
    ax.append(fig.add_subplot(1, 3, 2, label="Depth"))
    ax.append(fig.add_subplot(1, 3, 3, label="IR"))

    # Display images.
    im = []
    im.append(ax[0].imshow(image1.data))
    im.append(ax[1].imshow(image2.data, cmap='jet'))

    if len(cmap) == 0:
        im.append(ax[2].imshow(image3))
    else:
        im.append(ax[2].imshow(image3, cmap=cmap))


    # Create axes titles.
    ax[0].title.set_text('Color')
    ax[1].title.set_text('Depth')
    ax[2].title.set_text('Transformed Image')

    plt.show()

def auto_annotation():
    
    # Open a device using the "with" syntax.
    with k4a.Device.open() as device:

        # In order to start capturing frames, need to start the cameras.
        # The start_cameras() function requires a device configuration which
        # specifies the modes in which to put the color and depth cameras.
        # For convenience, the k4a package pre-defines some configurations
        # for common usage of the Azure Kinect device, but the user can
        # modify the values to set the device in their preferred modes.
        device_config = k4a.DeviceConfiguration(
            color_format=k4a.EImageFormat.COLOR_BGRA32,
            color_resolution=k4a.EColorResolution.RES_720P,
            depth_mode=k4a.EDepthMode.NFOV_2X2BINNED,
            camera_fps=k4a.EFramesPerSecond.FPS_15,
            synchronized_images_only=True
        )
        print(device_config)
        status = device.start_cameras(device_config)
        if status != k4a.EStatus.SUCCEEDED:
            raise IOError("Failed to start cameras.")

        # In order to create a Transformation class, we first need to get
        # a Calibration instance. Getting a calibration object needs the
        # depth mode and color camera resolution. Thankfully, this is part
        # of the device configuration used in the start_cameras() function.
        calibration = device.get_calibration(
            depth_mode=device_config.depth_mode,
            color_resolution=device_config.color_resolution)

        # Create a Transformation object using the calibration object as param.
        transform = k4a.Transformation(calibration)

        idx = 0
        # wait for Enter
        while True:
            input_str = input("Enterで撮影(F+Enterで終了)")
            if input_str == "f":
                break
            # Get a capture using the "with" syntax.
            with device.get_capture(-1) as capture:

                color = capture.color
                depth = capture.depth
                ir = capture.ir

                # Get a depth image but transformed in the color space.
                depth_transformed = transform.depth_image_to_color_camera(depth)
                depth_transformed_arr = depth_transformed.data
                mask = Image.open("mask.png")
                mask_arr = np.asarray(mask)
                depth_transformed_masked_arr = np.where(mask_arr == 0, 0, depth_transformed_arr)
                color_arr = color.data
                color_img = Image.fromarray(color_arr[:, :, [2, 1, 0, 3]])
                print(color_arr.shape)
                color_img.save(f'out{idx}_rgb.png')
                segmentation_img_arr = np.where(np.logical_and(depth_transformed_masked_arr < 905, depth_transformed_masked_arr != 0), 1, 0)
                segmentation_img = Image.fromarray(segmentation_img_arr.astype(np.uint8), "P")
                segmentation_img.putpalette([0,0,0,255,0,0])
                print(np.unique(depth_transformed_masked_arr))
                print("dtype")
                print(depth_transformed_masked_arr.dtype)
                depth_transformed_masked = Image.fromarray(depth_transformed_masked_arr)
                # segmentation_img.save(f'out{idx}_mask.png')
                segmentation_img.save(f'depth_mask.png')

                plot_images(color, depth, depth_transformed_masked, cmap='jet')
                idx += 1

        # There is no need to delete resources since Python will take care
        # of releasing resources in the objects' deleters. To explicitly
        # delete the images, capture, and device objects, call del on them.

if __name__ == '__main__':
    auto_annotation()