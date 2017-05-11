import numpy as np
import csv
import scipy.misc

def get_slices(data_file):
    # Get the features
    np_data = np.load(data_file)
    meta_reader = csv.reader(open(data_file.replace(".npy", ".csv"), "r"))
    n = len(np_data)

    slices = []
    meta = [next(meta_reader)]

    # Preprocess the features
    for k in range(n):
        tmp = np_data[k]
        next_csv = next(meta_reader)

        start = range(0, len(tmp) + 1, 50)
        stop = start[2:]
        L = [tmp[i : j] for i, j in zip(start, stop)]
        meta += [next_csv for i, j in zip(start, stop)]
        slices += L

    slices = np.array(slices)
    n = len(slices)

    #slices = slices.reshape(list(slices.shape) + [1]) 
    return n, slices, meta

def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """

    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data
