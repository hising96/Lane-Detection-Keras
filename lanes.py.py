import numpy as np
import cv2
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
model = keras.models.load_model('model.h5')

class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def road_lines(image):
    
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    prediction = model.predict(small_img)[0] * 255

    lanes.recent_fit.append(prediction)
   
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

   
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    lane_image = imresize(lane_drawn, (720, 1280, 3))
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result

lanes = Lanes()
vid_input = VideoFileClip("rain.mp4")
vid_output = 'rain_output.mp4'

vid_clip = clip1.fl_image(road_lines)
vid_clip.write_videofile(vid_output)
