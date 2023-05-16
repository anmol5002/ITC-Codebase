import elementpath                                              
import xml.etree.ElementTree as ET                              
import tensorflow as tf
import os                               #act as operating system and automate the process         #
import re                                                                                         # Basic libraries are imported for different tasks                            
import cv2 as cv                        # used for basic image cropping and saving                #
from harvesters.core import Harvester   # used for image capturing                                #
import atexit
import time                             # check time of each operation


camera_id = '700004978993'               # cam 3 id
iterator = 0

save_file_path = r'C:\Users\itc.DESKTOP-RF2G1DL\Desktop\i24062020'       # file_path where images will be saved

if not os.path.exists(save_file_path):                                   # if that kind of file_path not available make a folder automatically
    os.mkdir(save_file_path)


def close_camera(ia):                                                    # function to close camera capturing images
    if ia != None:
        ia.stop_image_acquisition()

try:
    ia = None
    atexit.register(close_camera, ia=ia)

   
    def image_saver(img):                                                 # function to save images
        global iterator
        image_path = os.path.join(save_file_path, str(iterator) + '.jpg')
        cv.imwrite(image_path, img)
        iterator += 1
        return img

    def initCamera(set_dimensions, ia):                                   # function specifying basic variables of image captured via camera
        ia.device.node_map.PixelFormat.value = 'BayerRG8'
        ia.device.node_map.Gain.value =45                                 # increasing gain will increase brightness and color contrast
        ia.device.node_map.ExposureTime.value = 700
        ia.device.node_map.TriggerMode.value = 'On'
        ia.device.node_map.TriggerDelay.value = 100000
        if set_dimensions:
            ia.device.node_map.Width = 992                                # This specifies width of image captured 
            ia.device.node_map.Height = 600                               # This specifies height of image captured
            ia.device.node_map.OffsetX = 528                              # Image captured by camera is big ( so this specifies how many pixels to reject )
            ia.device.node_map.OffsetY = 382                              # Similarly
        ia.start_image_acquisition()
        return ia

    print("working")

    h = Harvester()
    h.add_cti_file(r"bgapi2_gige.cti")                                    # Camera file used for capturing and needs to be put at same location as this code
    h.update_device_info_list()
    selected_cam = -1
    for cam in h.device_info_list:
        if cam.serial_number == camera_id:
            selected_cam = h.device_info_list.index(cam)
    if selected_cam > -1:
        ia = h.create_image_acquirer(selected_cam)
    else:
        input('Camera not connected. Press Enter to exit.')               # If any camera issue, this statement will be printed on dashboard
        exit()

    try:
        ia = initCamera(True, ia)                                         # Function named initCamera is called and now has initialised all constant variables
    except:
        if ia is not None:
            ia.stop_image_acquisition()
        ia = initCamera(False, ia)
    image = None

    print('working 2')

    def return_img_numpy_array():                                          # This function will return an image of size 992*600 pixels
        global ia
        global time_start
        global image
        with ia.fetch_buffer() as buffer:
            if len(buffer.payload.components) > 0:
                component = buffer.payload.components[0]

                image = component.data.reshape(
                    component.height, component.width
                )
                image = cv.cvtColor(image, cv.COLOR_BAYER_RG2RGB)
                ##                scale_percent = 100  # percent of original size
                ##                width = int(image.shape[1] * scale_percent / 100)
                ##                height = int(image.shape[0] * scale_percent / 100)
                dim = (992, 600)
                resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)        
                return resized
            return None

    while True:                                                   # This is an infinite loop which will remain active till code is not terminated
        image = return_img_numpy_array()
        if str(type(image)) == "<class 'numpy.ndarray'>":
            vis = image_saver(image)                              # Here, image is saved at desired location
            cv.imshow("Quality Inspection Top Side", vis)         # This block shows a preview of each image captured
            print("1")
            if cv.waitKey(1) == 27:
                break

    ia.stop_image_acquisition()

except Exception as e:                                            # If any exception occurs at any stage, this code triggers and shows error statement
    if ia is not None:
        ia.stop_image_acquisition()
    print("Exception", str(e))
    input('press enter to exit')

exit(0)                                                           # after enter is pressed, code will terminate
