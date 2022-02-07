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
        img = np.array(Image.open(f))
        if len(img.shape) == 3:
            img = img.mean(axis=2)
        image_stack.append(img.astype(np.float32))
    return image_stack, files


VN_masks = {
    4: [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    8: [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    12: [[0, 0, 1, 0, 0],
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


def calc_OP_weights(img, target, OP_curve, inverse):
    """
    Calculate the out of plane weights for each pixel based off of its value
    compared to a target value and the weighting equation parameters

    Parameters
    ----------
    %(input)
    img : 2D numpy array
        The array for which per-pixel weights are being determined
    target : float
        The value which gets the strongest connection to the source during
        the graph cut.
    OP_curve : float
        the "sharpness" of the weighting curve. higher means less exclusive
    OP_strength : float
        multiplier for increasing the weights of the OP connections
        relative to the IP connections
    inverse : bool
        if true, sink weights will be 1/source_weights. if false, sink weights
        will be 1 - source_weights. These are two common alternative methods
        for weighting OP connections, the division one is better for sharper
        cuts in images with low gradients accross a given feature.

    %(output)
    source : 2D numpy arrays
        source weights
    source : 2D numpy arrays
        sink weights
    """
    likelyhood = np.abs(img - target)
    likelyhood = likelyhood/likelyhood.max()
    OP_weight = (likelyhood*0 + 1) - (likelyhood**OP_curve)
    if inverse:
        source = OP_weight + 0.001
        sink = 1/source
    else:
        source = OP_weight*1
        sink = 1-source
    return(source, sink)


def calc_IP_weights(img, IP_curve, cutoff, IP_min, IP_nn, style):
    """
    Calculate the in plane weights for each pixel based off the weighting
    equation parameters.

    Parameters
    ----------
    %(input)
    img : 2D numpy array
        The array for which per-pixel weights are being determined
    IP_curve : float
        the "sharpness" of the weighting curve. higher means less exclusive
    cutoff : float
        intra-pixel deltas larger than this get the min IP weighting
    IP_min : float
        minimum IP connection weight, as a fraction of the maximum
    IP_nn : int
        either 4 or 8, number of nearest neighbors to consider when
        calculating in plane connections
    style : str
        either "sobel" or delta right now. determines how edge intensities are
        chosen.
    %(output)
    IP_weights : list of 2D numpy arrays
        weights that should be applied for a given connection direction
    neighbor_map : list of 3x3 numpy arrays
        maps for the connection directions in which the IP weights should be
        applied.
    """
    if style == 'delta':
        LR_delta = np.zeros(img.shape)
        LR_delta[:, :-1] = np.abs(img[:, 1:] - img[:, :-1])
        UD_delta = np.zeros(img.shape)
        UD_delta[:-1, :] = np.abs(img[1:, :] - img[:-1, :])
        D1_delta = np.zeros(img.shape)
        D1_delta[:-1, :-1] = np.abs(img[1:, 1:] - img[:-1, :-1])
        D2_delta = np.zeros(img.shape)
        D2_delta[:-1, 1:] = np.abs(img[1:, :-1] - img[:-1, 1:])
    elif style == 'sobel':
        LR_delta = ndimage.sobel(img, axis=0)
        UD_delta = ndimage.sobel(img, axis=1)
        D1_delta = np.abs(np.hypot(LR_delta, UD_delta))
        D2_delta = np.abs(np.hypot(LR_delta, UD_delta*-1))
        LR_delta = np.abs(LR_delta)
        UD_delta = np.abs(UD_delta)
    else:
        NotImplementedError
    raw_weights = [(cutoff - x)/cutoff for x in
                   [LR_delta, UD_delta, D1_delta, D2_delta]]
    normed_weights = [(x > 0)*x for x in raw_weights]
    eqn_weights = [1 - (x**IP_curve) for x in normed_weights]
    final_weights = [(x > IP_min)*(x - IP_min) + IP_min for x in eqn_weights]
    if IP_nn == 8:
        return ([final_weights],
                [VN_locs[x] for x in ['R', 'D', 'UR', 'DR']])
    else:
        return (final_weights[:2],
                [VN_locs[x] for x in ['R', 'D']])


def dialate_mask(sgm, dialation_steps, dialation_nn):
    """
    Dialate the image
    Its late and im tired. put better description later
    """
    mask = VN_masks[dialation_nn]
    if dialation_steps >= 2:
        dialated = sgm
        for i in np.arange(dialation_steps):
            dialated = ndimage.binary_dilation(dialated, mask)
    elif dialation_steps >= 1:
        dialated = ndimage.binary_dilation(sgm, mask)
    else:
        dialated = sgm
    return dialated


def ff_clean_and_label(img, min_spot_size, nn):
    """
    clean_and_label the image
    Its late and im tired. put better description later
    """
    mask = VN_masks[nn]
    # segment boolean array
    init_label = ndimage.label(img, mask)[0]
    # get a count of how big each segmented object is
    ID, count = np.unique(init_label, return_counts=True)
    # make a dictionary to filter out grains below minimum size threshold
    large_ID = ID*(count > min_spot_size)
    big_filter = dict(zip(ID, large_ID))
    final_label = np.vectorize(big_filter.__getitem__)(init_label)
    # Dialate every spot twice so you get a 2 pixel buffer
    dialate1 = ndimage.binary_dilation(final_label > 0, mask)
    dialate2 = ndimage.binary_dilation(dialate1, mask)
    # Now relabel them so every spot has its own ID.
    labeled_spots = ndimage.label(dialate2, mask)[0]
    return(labeled_spots)
