# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import segmentation as skseg
# import networkx as nx
import maxflow
from scipy import ndimage


# helper functions
def make_a_guess(img, past_guesses):
    med_img = ndimage.median_filter(img,size=2)
    choices = med_img[med_img >= 0]
    count = np.histogram(choices, bins=np.arange(-0.5, 256.5))[0]
    count[past_guesses] = 0
    if np.sum(count) == 0:
        past_guesses = []
        count = (np.histogram(choices, bins=np.arange(-0.5, 256.5))[0])**3
        print("      BEANS!!!!")
    count = count/np.linalg.norm(count)
    choice = np.random.choice(np.arange(256), 1, p=(count.astype(float))**2)[0]
#    choice = choices[np.random.randint(choices.size)]
    past_guesses.append(choice)
    return choice, past_guesses


orig_img = np.array(Image.open('median.tiff'), dtype='int16')
# Right off the bat, give the background a special unique color -1. from now
# on, things with negative values are already segmented
img = skseg.flood_fill(orig_img, (0, 0), -1)
# now make the big master graph for the cutting. we will slice off pieces of
# this as we go through the calculations

# img = np.array(Image.open('../median.tiff'), dtype='int16')
# img = img[:500, :500]
plt.close('all')
plt.figure()
plt.imshow(orig_img)
# Make the data for the directed graph that WONT change during the calculations
N = np.size(img)
# goodish setting: 0.8, 100, 20
# badish/goodish setting: 0.5,100,1
curve = 0.8
cutoff = 100
LR_delta = np.abs(img[:, 1:] - img[:, :-1])
UD_delta = np.abs(img[1:, :] - img[-1:, :])
LR_weight = (cutoff**curve) - (LR_delta**curve)
UD_weight = (cutoff**curve) - (UD_delta**curve)
LR_weight[LR_weight <= 20] = 20
UD_weight[UD_weight <= 20] = 20
# LR_weight = (256-np.abs(img[:, 1:] - img[:, :-1]))**1.4
# UD_weight = (256-np.abs(img[1:, :] - img[:-1, :]))**1.4
LR_weight = np.hstack([LR_weight, np.zeros([img.shape[0], 1])])*1
UD_weight = np.vstack([UD_weight, np.zeros([1, img.shape[1]])])*1
LR_struct = [[0, 0, 0],
             [0, 0, 1],
             [0, 0, 0]]
UD_struct = [[0, 0, 0],
             [0, 0, 0],
             [0, 1, 0]]

plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(LR_weight[100:350, 900:1400])
plt.figure()
plt.imshow(img[100:350 , 900:1400])
# Set up for doing cuts
iteration = 0
past_guesses = []
while np.sum(img > 0) > 100:
    iteration += 1
    # make your directed graph
    g = maxflow.GraphFloat(N, N*4)
    nodeids = g.add_grid_nodes(img.shape)
    aaa = nodeids*1
    # Ignore nodes that have already been assigned to grains
    nodeids[img < 0] = -1
    # now add in-plane edge weights (LR first, then UD)
    g.add_grid_edges(nodeids, LR_weight, LR_struct, symmetric=True)
    g.add_grid_edges(nodeids, UD_weight, UD_struct, symmetric=True)
    # Pick a greyscale value from the unassigned area, and calculate per-voxel
    # Likelyhoods based on how close they are to that value
    guess, past_guesses = make_a_guess(img, past_guesses)
    likelyhood = 10*(((256-np.abs(img-guess))/256)**20)
    likelyhood[likelyhood <= 0.01] = 0.01
    likelyhood[img<=0] = 0
    # Add those likelyhoods as terminal edge weights (ie, out of plane weights)
    g.add_grid_tedges(nodeids, 10.1-likelyhood, likelyhood*(1.02**iteration))
    # Graph is complete. Do the maxflow and grab the "pulled out" stuff
    g.maxflow()
    print(guess)
    sgm = g.get_grid_segments(np.arange(N).reshape(img.shape))
    if np.sum(sgm) > 100:
#        plt.imshow(likelyhood)
        # uncomment this stuff if you wanna see what is happening here
#        plt.figure()
#        plt.imshow(sgm)

    # graph cut can occasionally grab several grains per cut, including some
    # tiny grains it really shouldn't. To prevent that, lets filter them out.
        min_grain_size = 3
        out = ndimage.label(sgm, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])[0]
        ID, count = np.unique(out, return_counts=True)
        large_ID = ID*(count > min_grain_size)
    # large_ID are JUST the IDs of grains bigger than min_grain_size. now we do
    # some python-foo to change that large_ID list to a sequential list, then
    # apply it to the "out" array to give a final labeled array
        old_to_new_id = dict(zip(ID[count > 1], np.arange(len(ID[count > 1]))))
        new_ID = np.vectorize(old_to_new_id.__getitem__)(large_ID)
        translator = dict(zip(ID, new_ID))
        labeled_grains = np.vectorize(translator.__getitem__)(out)
#        plt.figure()
#        plt.imshow(labeled_grains)
    # Now, finally, take all these new labels, and apply them as NEGATIVE
    # values to the img object. This way, positive values are greyscale values
    # from the unsegemented areas of the map, and negative values are grain_IDs
        new_labels = img.min()+(labeled_grains[labeled_grains > 0] * -1)
        img[labeled_grains > 0] = new_labels
        print(np.sum(img >= 0), np.size(np.unique(img[img <= 0])))
#        plt.figure()
#        plt.imshow(img)
    else:
        aaaa = 1
        #print('beans!')
    if iteration >2:
        img[img >= 0] = 0

    # uncomment this stuff if you wanna see what is happening here
plt.figure()
plt.imshow(np.stack([img, img/7, img/23], axis=2) % 255 / 255)