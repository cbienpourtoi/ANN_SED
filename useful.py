__author__ = 'loic'

import numpy as np

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def nmad(arr):
    """ Normalized mad (so that it has the same scale than a sigma)
    https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation
    """
    return 1.4826 * mad(arr)

def outliers(diff, specz, deviation):
    """ Outputs the percentage of outliers
    """
    nOut = np.count_nonzero((np.abs(diff)/(1+specz)) > (5. * deviation))
    percentOut = nOut / float(len(diff)) * 100.
    return percentOut