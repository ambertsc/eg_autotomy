import os

import numpy as np
import scipy
from scipy.ndimage import label

def check_connected(body):

    # check if body plan (np.array) is connected
    #

    labels = label(body)[0]

    if labels.max() > 1:
        return False
    else:
        return True

