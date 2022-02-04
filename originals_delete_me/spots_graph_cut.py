# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:05:41 2022

@author: agerlt

# ===================== #
# NOTE TO MATLAB USERS:
Nothing in this code explicitly requires python, and recreating this in MATLAB
should be doable. The magic function you need is called "digraph", and there is
a maxflow function in one of the toolboxes. HOWEVER, the process of building
the graph is fundamentally different, so don't try to make a line-for-line copy
as it will certainly fail. instead, just google "max cut min flow image
processing MATLAB" and go from there.
# ===================== #
"""

import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import maxflow
from scipy import ndimage


# ========================================================================= #
# INPUTS
filenames = "LabHEDM_*.tif"
curve = 0.8
high_threshold = 150
low_threshold = 5
cutoff = 100
noise_threshold = 20
min_dialated_spot_size = 9  # Don't go below 9, things will break
# ========================================================================= #

# plt.close('all')
# check that there are files matching the filename search text
assert len(glob.glob(filenames)) > 0, "no files found matching the search term"
# load up all the images into a list of numpy arrays
images = [np.array(Image.open(img))[:, :, 0]for img in glob.glob(filenames)]

# Create a graph constructor variables that are image independent
LR_struct = [[0, 0, 0],
             [0, 0, 1],
             [0, 0, 0]]
UD_struct = [[0, 0, 0],
             [0, 0, 0],
             [0, 1, 0]]
# we don't use these yet ,but might in the future, so leaving for now
D5_struct = [[0, 0, 1],
             [0, 0, 0],
             [0, 0, 0]]
D7_struct = [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 1]]
mask_8 = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
mask_4 = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

for i_count, img in enumerate(images):
    print("graph cutting {} ...".format(glob.glob(filenames)[i_count]))
    IP_img = img*1
    IP_img[IP_img < low_threshold] = low_threshold
    IP_img[IP_img > high_threshold] = high_threshold

    # Find the Out-of-plane weights
    print("Generating graph weights")
    likelyhood = (IP_img - low_threshold)/(high_threshold-low_threshold)
    OP_wts = (likelyhood**0.5)*(cutoff**curve)*10

    # Find the in-plane weights
    LR_delta = np.abs(IP_img[:, 1:] - IP_img[:, :-1])
    UD_delta = np.abs(IP_img[1:, :] - IP_img[-1:, :])
    LR_weight = (cutoff**curve) - (LR_delta**curve)
    UD_weight = (cutoff**curve) - (UD_delta**curve)
    LR_weight[LR_weight <= noise_threshold] = noise_threshold
    UD_weight[UD_weight <= noise_threshold] = noise_threshold
    LR_weight = np.hstack([LR_weight, np.zeros([img.shape[0], 1])])
    UD_weight = np.vstack([UD_weight, np.zeros([1, img.shape[1]])])
    LR_weight = LR_weight*(1-likelyhood)
    UD_weight = UD_weight*(1-likelyhood)

    # make the actual graph and remove obviously bad pixels
    print("Building the graph")
    N = img.size
    g = maxflow.GraphFloat(N, N*4)
    nodeids = g.add_grid_nodes(img.shape)
    nodeids[img < low_threshold] = -1
    # add the edges (ignoring nodes below threshold for speed)
    g.add_grid_tedges(nodeids, OP_wts.max() + 1 - OP_wts, OP_wts)
    g.add_grid_edges(nodeids, LR_weight, LR_struct, symmetric=True)
    g.add_grid_edges(nodeids, UD_weight, UD_struct, symmetric=True)

    # Graph is complete. Do the maxflow and grab the "pulled out" stuff
    print("Computing the cut")
    g.maxflow()
    sgm = g.get_grid_segments(np.arange(N).reshape(img.shape))

    # This result is okay, but not great. Lets clean it up.
    print("Post Processing")
    # First, dialate all the spots to connect same-spot regions
    dialated = ndimage.binary_dilation(sgm, mask_8)
    # Remove all the small spots that didn't have nearby neghbors
    labeled = ndimage.label(sgm, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])[0]
    ID, count = np.unique(labeled, return_counts=True)
    large_ID = ID*(count > min_dialated_spot_size)
    old_to_new_id = dict(zip(ID[count > 1], np.arange(len(ID[count > 1]))))
    new_ID = np.vectorize(old_to_new_id.__getitem__)(large_ID)
    translator = dict(zip(ID, new_ID))
    good_spots = np.vectorize(translator.__getitem__)(labeled)
    # Dialate things again twice so you get a 2 pixel buffer around every spot
    dialated_spot_map = ndimage.binary_dilation(good_spots, mask_8)
    dialated_spot_map = ndimage.binary_dilation(dialated_spot_map, mask_8)
    # Now relabel them so every spot has its own ID.
    good_labeled_spots = ndimage.label(dialated_spot_map, mask_4)[0]

    # Now a couple graphs to see what we did.
    plt.figure("Original")
    plt.imshow(img)  # [2130:2200,3250:3320])
    plt.figure("IP_weights")
    plt.imshow(-1*(LR_weight + UD_weight))
    plt.figure("OP_weights")
    plt.imshow(OP_wts)
    plt.figure("filtered")
    plt.imshow((img*(dialated_spot_map > 0)))
    plt.figure("Segmented")
    plt.imshow(good_labeled_spots)

    # Save out those images as well
    plt.imsave("Original_{}.tiff".format(i_count), img)
    plt.imsave("IP_weights_{}.tiff".format(i_count),
               256 - (1*(LR_weight + UD_weight)))
    plt.imsave("OP_weights_{}.tiff".format(i_count), OP_wts)
    plt.imsave("Filtered_{}.tiff".format(i_count),
               img*(dialated_spot_map > 0))
    plt.imsave("Segmented_{}.tiff".format(i_count), good_labeled_spots)

    print("Donezo!!!")