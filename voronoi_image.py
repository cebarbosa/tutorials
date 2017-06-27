# -*- coding: utf-8 -*-
"""

Created on 27/06/2017

@Author: Carlos Eduardo Barbosa

Example of Voronoi binning.

"""

import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf

###############################################################################
# You can download the Voronoi package at Michelle Cappellari website at
# http://www-astro.physics.ox.ac.uk/~mxc/software/#binning
# To use as a package without need of installation
# (as in this example) add an empty text file called
# __init__.py in the voronoi directory
from voronoi.voronoi_2d_binning import voronoi_2d_binning
##############################################################################

def example():
    """ Example of Voronoi binning """
    signal = pf.getdata("./images/ugc8334.fits") # Image already sky-subtracted
    mask = pf.getdata("./images/mask.fits") # Mask for the image
    mask[:20,:] = 1 # Additional masking in the bottom of figure
    # Rebin image for sake of speed
    dim = 128 # has to be a power of 2, maximum of 1024
    signal = bin_ndarray(signal, (dim, dim))
    mask = bin_ndarray(mask, (dim, dim))
    # Masking image before tesselation
    signal = np.where(mask == 0, signal, np.nan)
    # Noise image assuming Poisson shot noise and sky sigma of 100 ADU
    noise = np.sqrt(np.clip(signal, 0, np.infty) + 100**2)
    # Setting the S/N per bin
    targetSN = 60
    # Setting up auxiliary arrays
    goodpix = np.isfinite(signal)
    ydim, xdim = signal.shape
    xaxis = np.arange(1, xdim+1)
    yaxis = np.arange(1, ydim+1)
    xx, yy = np.meshgrid(xaxis, yaxis) # 2D arrays of position
    ###########################################################################
    # Running the Voronoi binning
    classe = voronoi_2d_binning(xx[goodpix], yy[goodpix], signal[goodpix],
                               noise[goodpix], targetSN, plot=0,
                               quiet=0, pixelsize=1, cvt=False)[0] # <-- only first output is retrieved
    # See documentation in the voronoi file for explanation of all outputs
    # We only need the vector 'classe', which indicates the bin associated
    # to each pixel, that is why we are only retrieving the first output above
    ###########################################################################
    # First example: visualize the bins
    plt.figure(1)
    bins = np.zeros_like(signal) * np.nan
    bins[goodpix] = classe
    plt.imshow(bins, origin="bottom", cmap="rainbow")
    ##########################################################################
    # Second example: averaging within bins
    plt.figure(2, figsize=(8,4))
    signal_binned = np.zeros_like(signal) * np.nan
    for bin in np.unique(classe):
        idx = np.where(bins==bin)
        signal_binned[idx] = signal[idx].mean()
    ax = plt.subplot(121)
    ax.set_title("Original")
    ax.set_aspect("equal")
    ax.imshow(signal, origin="bottom", cmap="cubehelix", vmin=200,
              vmax=8000)
    ax = plt.subplot(122)
    ax.set_title("Average Voronoi binning")
    ax.set_aspect("equal")
    ax.imshow(signal_binned, origin="bottom", cmap="cubehelix", vmin=200, vmax=8000)

    plt.show()


def bin_ndarray(ndarray, new_shape, operation='mean'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Source: https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray


if __name__ == "__main__":
    example()