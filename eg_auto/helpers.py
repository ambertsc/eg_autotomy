import os
import time

import numpy as np
import scipy
from scipy.ndimage import label

import PIL
from PIL import Image

def check_connected(body):

    # check if body plan (np.array) is connected
    #

    labels = label(body)[0]

    if labels.max() > 1:
        return False
    else:
        return True

def make_gif(frames_path="./frames/", gif_path="./assets", \
        tag="no_tag", speedup=3, scale=1.0):
    
    dir_list = os.listdir(frames_path)

    frames = []

    dir_list.sort()
    for ii, filename in enumerate(dir_list):
   
        if "png" in filename and (ii % speedup) == 0:

            image_path = os.path.join(frames_path, filename)
            frames.append(Image.open(image_path))

    assert len(frames) > 1, "no frames to make gif"

    first_frame = frames[0]
    
    gif_id = int((time.time() % 1)*1000)

    gif_path = os.path.join(gif_path, f"gif_{tag}_{gif_id:04d}_{speedup}X.gif") 

    first_frame.save(gif_path, format="GIF", append_images=frames, \
            save_all=True, duration=42, loop=0)

    rm_path = os.path.join(frames_path, "*png")

    os.system(f"rm {rm_path}")

