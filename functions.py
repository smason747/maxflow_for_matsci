# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:11:03 2022

@author: agerlt
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
from glob import glob as glob
from PIL import Image

def load_ang(ebsd_file):
    raise NotImplementedError
    return


def load_images(fnames):
    """load an image or a stack of images from files

    Parameters
    ----------
    %(input)
    fnames : str or list of str
        either a filename or a list of filenames. can also accept wildcard
        statements (such as "*.tif") similar to how ls works in terminal/.
    %(output)
    images : list of 2D numpy arrays
        list of float32 single channel 2D arrays.

    Notes
    ------
    - If data is stored in multiple channels (such as RGB data), that will be
      lost by this import function. if you would like to have the graph cut
      consider data in different channels differently, you will need to write
      your own graph cut algorithm.
    """

    if type(fnames) == list:
        files = np.unique(np.concatenate([glob(x) for x in fnames])).tolist()
    elif type(fnames) == str:
        files = np.unique(glob(fnames)).tolist()
        fnames = [fnames]
    assert type(files) == list, """
    'fnames' must be interpretable as a filename or list of filenames"""
    assert len(files) > 0, """
    no files found matching the provided filename list:
        \n - {}""".format('\n - '.join(fnames))

    image_stack = []
    for f in files:
        img = Image.open(f)
        if len(img.shape) == 3:
            img = img.mean(axis=2)
        image_stack.append(img.astype(np.float32))
    return image_stack, files


VN_masks = {
    1: [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    2: [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    3: [[0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]]
}

VN_locs = {
    "UR": [[0, 0, 1],
           [0, 0, 0],
           [0, 0, 0]],
    "R":  [[0, 0, 0],
           [0, 0, 1],
           [0, 0, 0]],
    "DR": [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 1]],
    "D":  [[0, 0, 0],
           [0, 0, 0],
           [0, 1, 0]],
    "RR":  [[0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]],
    "DD":  [[0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0]]
}
