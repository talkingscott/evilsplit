"""
Data augmentation.
"""

import numpy as np

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28

def noop_augment(multiple, ins, outs):
    """
    Performs no data augmentation.
    """
    print(type(ins))
    print(ins.shape)
    print(type(outs))
    print(outs.shape)
    return ins, outs

def simple_augment(multiple, ins, outs):
    """
    Simple, naive data augmentation that allows a pixel's neighbors to bleed
    into or from it.
    """
    print(type(ins))
    print(ins.shape)
    print(type(outs))
    print(outs.shape)

    new_ins = np.array(ins, copy=True)
    new_outs = np.array(outs, copy=True)
    for m in range(multiple):
        new_ins = np.concatenate((new_ins, ins))
        new_outs = np.concatenate((new_outs, outs))

    # certainly this can be done more efficiently
    for i in range(ins.shape[0]):
        if i % 1000 == 999:
            print('Augment {}'.format(i + 1))

        for p in range(ins.shape[1]):
            neighbors = []
            above = p - IMAGE_WIDTH
            if above >= 0:
                neighbors.append(ins[i, above])
            if (p % IMAGE_WIDTH) != 0:
                left = p - 1
                neighbors.append(ins[i, left])
            if (p % IMAGE_WIDTH) != (IMAGE_WIDTH - 1):
                right = p + 1
                neighbors.append(ins[i, right])
            below = p + IMAGE_WIDTH
            if below < (IMAGE_HEIGHT * IMAGE_WIDTH):
                neighbors.append(ins[i, below])

            this_pixel = ins[i, p]
            neighbor_pixels = np.mean(neighbors)

            baseline = min(this_pixel, neighbor_pixels)
            difference = abs(this_pixel - neighbor_pixels)

            if difference == 0.0:
                # this pixel and its neighbors are in equillibrium, can't bleed
                continue

            for m in range(multiple):
                new_ins[(ins.shape[0] * (m + 1)) + i, p] = np.random.uniform(baseline, baseline + difference)

    print(new_ins.shape)
    print(new_outs.shape)

    return new_ins, new_outs
