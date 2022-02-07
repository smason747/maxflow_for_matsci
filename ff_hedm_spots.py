# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:14:55 2022

@author: agerlt

Example on how to use graph cut to pull spots from a ff-HEDM scan.
"""

import numpy as np
import matplotlib.pyplot as plt
import maxflow  # NOTE, "pip install pymaxflow", NOT "pip install maxflow"
import functions as fn
from PIL import Image
import os

# ========================================================================= #
# USER INPUTS
# these are the default values used in the graph cutting algorithm. they are
# by no means optimal or even good, they are just the ones that I got to work
# when testing this concept out. Users are encouraged to modify them and watch
# how the cuts change

filenames = "Data/ff_*.tif*"  # text or list of text; accepts wildcards
high_threshold = 200  # intensities above this are given this value
low_threshold = 5  # intensities below this are given this value

# In-plane (IP) weighting (how much pixels want to group with their neighbors)
IP_curve = 0.8  # how "sharp" the weighting curve is. higher = more exclusive
cutoff = 100  # intra-pixel deltas larger than this get the min IP weighting
IP_min = 0.2  # minimum IP connection weight, as a fraction of the maximum
IP_nn = 4  # number of nearest neighbors to consider for IP connections

# Out-of-plane (OP) weighting (how strongly pixels are pulled on during cut)
OP_curve = 10
OP_strength = 1
inverse = True

# Post Processing
dialation_steps = 2  # number of times result is dialated before segmentation
dialation_nn = 4  # number of nearest neighbors to consider for dialation
min_spot_size = 36  # Don't go below 9, things will break

# Saving data
save_as_img = False
save_as_txt = False

# visualization tools (switch off for large processes)
show_weighting_equations = True
show_IP_map = True
show_OP_map = True
show_original = True
show_pre_clean = True
show_filtered = True
show_labeled = True
# ========================================================================= #


#plt.close('all')

# load up all the files from filenames as a list of float32 numpy arrays
image_stack, files = fn.load_images(filenames)

# loop through every image in the stack and perform the graph cut
for i_count, orig_img in enumerate(image_stack):
    # Only need to do a single graph cut to seperate out the spots.
    print("\n\n{}\n working on image {}\n".format("="*40, files[i_count]))

    # Pre-process image according to thresholds
    img = orig_img*1
    img[img < low_threshold] = low_threshold
    img[img > high_threshold] = high_threshold

    # Make an empty graph,then exclude any pixels too dark to even consider.
    # NOTE: don't worry if we take out a few important oddball pixels here,
    # we are going to do a dialation step at the end to regrab them.
    N = img.size
    g = maxflow.GraphFloat(N, N*4)
    nodeids = g.add_grid_nodes(img.shape)
    nodeids[img < low_threshold] = -1

    print("Building the graph...")
    # Add the Out-of-Plane weights. since IP weights are all auto-weighted to
    # max out at one, increasing or reducing these weights will control how
    # strongly the source and sink can "pull" a pixel to them versus the pull
    # felt from the In-Plane connections
    source, sink = fn.calc_OP_weights(img, high_threshold, OP_curve, inverse)
    # apply the weights
    g.add_grid_tedges(nodeids, sink*OP_strength, source*OP_strength)

    # Add the In-Plane weights. All connections are  between 0 and 1.
    # Two notes:
    # 1) If users want to try an initial median filter to remove noise, this
    #    Is where to insert that. Personally, I found no improvement and maybe
    #    even harm in adding one, but maybe other images will be different
    # 2) Here, the IP weights are calculated from the delta between neighboring
    #    pixels, but a sobel filter is also common. Change "style" to "sobel"
    #    see how the results change, and/or add your own.
    IP_ws, neigh_map = fn.calc_IP_weights(img, IP_curve, cutoff,
                                          IP_min, IP_nn, style='sobel')
    # Extra IP_weighting step specifically for ff_HEDM data. I found that if
    # I took the IP connections, multiplied them by their intensity, then
    # renormalized, I got much better results (IE, neighbors that are similar
    # AND have a high luminosity have stronger bonds than background pixels).
    # specifically, this did a good job of not accidentally dragging along
    # noisy background areas. It DID drop a few parts of the actual spots, but
    # we get them back during the dialation phase.
    likelyhood = (img - low_threshold)/(high_threshold-low_threshold)
    likelyhood[likelyhood<0.5] = 0.5
    IP_ws = [x*likelyhood for x in IP_ws]
    IP_ws = [x/(x.max()) for x in IP_ws]
    # apply the weights
    for neigh, IP_w in enumerate(IP_ws):
        g.add_grid_edges(nodeids, IP_w, neigh, symmetric=True)

    print("Computing the cut...")
    g.maxflow()  # does the actual cut, saves edge capacities as part of graph
    sgm_in = np.arange(N).reshape(img.shape)
    sgm = g.get_grid_segments(sgm_in)  # boolean map of the foreground

    print("Post Processing the cut...")
    # This result is okay, but not great. We need to clean them up some.
    # First, dialate all the spots to connect regions
    dialated = fn.dialate_mask(sgm, dialation_steps, dialation_nn)

    # Remove spots below the minimum threshold and assign labels
    good_spots = fn.ff_clean_and_label(dialated, min_spot_size, dialation_nn)
    filtered = orig_img*(good_spots != 0)

    print("Saving and visualization...")
    # at this point, save any outputs the user requested to the Output folder
    sname = "".join(files[0].split(os.sep)[-1].split(".")[:-1])
    savename = "Data/Output/" + sname
    if save_as_img:
        Image.fromarray(filtered).save(savename + "_out.tif")
        Image.fromarray(good_spots).save(savename + "_labeled.tif")
    if save_as_txt:
        np.savetxt(savename + "_out.txt", filtered)
        np.savetxt(savename + "labeled.txt", good_spots)

# ================================================== #
# Graphing stuff (no calculations, can be ignored)
# ================================================== #
    # add some graphs users can turn on and off to help with comprehension
    if show_weighting_equations:
        fig_name = "img{} IP and OP weighting equations".format(i_count+1)
        diff = high_threshold - low_threshold
        hi_lo = np.arange(0, diff, 0.01)
        hl_source, hl_sink = fn.calc_OP_weights(hi_lo, 150, OP_curve, inverse)
        IP_line = (hi_lo*0 + diff)**IP_curve - (hi_lo**IP_curve)
        IP_line = IP_line/IP_line.max()
        fig, ax = plt.subplots(num=fig_name)
        l1, = ax.plot(hi_lo, IP_line, 'b', label='In-Plane weights')
        l2, = ax.plot(hi_lo, hl_source, 'r', label='source weights')
        l3, = ax.plot(hi_lo, hl_sink, 'g', label='sink weights')
        ax.grid()
        ax.set_xlim(-0.01, diff+0.01)
        ax.set_ylim(-0.01, np.max([source.max(), IP_line.max()])+0.01)
        ax.legend(handles=[l1, l2, l3])
    if show_IP_map:
        plt.figure("In-Plane connection map (averages)")
        plt.imshow(np.stack(IP_ws, axis=2).mean(axis=2))
#        plt.imsave(savename+"_IP.png", IP_ws)
    if show_OP_map:
        plt.figure("Out-Of-Plane connection map")
        plt.imshow(source)  # [2050:2350,3200:3500])
#        plt.imsave(savename+"_OP.png", source)
    if show_original:
        plt.figure("Original Image")
        plt.imshow(orig_img)  # [2050:2350,3200:3500])
#        plt.imsave(savename+"_orig.png", orig_img)
    if show_pre_clean:
        plt.figure("Image before Post")
        plt.imshow(sgm)  # [2050:2350,3200:3500])
#        plt.imsave(savename+"_pre_clean.png", orig_img)
    if show_filtered:
        plt.figure("original spots with background removed")
        plt.imshow(filtered)  # [2050:2350,3200:3500])
    if show_labeled:
        plt.figure("labeled spots")
        plt.imshow(good_spots)  # [2050:2350,3200:3500])

    print(files[i_count]+" Complete")
print("DONEZO!")
