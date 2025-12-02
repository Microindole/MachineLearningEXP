########################################################################
#
# Module for caching data-structures to files.
#
# Implemented in Python 3.5
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import os
import pickle
import numpy as np

def cache(cache_path, fn, *args, **kwargs):
    """
    Cache-wrapper for a function or class-method.

    So if the cache-file already exists then the data is reloaded
    and returned, otherwise the function is called and the result
    is saved to cache.

    :param cache_path:
        File-path for the cache-file.

    :param fn:
        Function or class-method to be called.

    :param args:
        Arguments to the function or class-method.

    :param kwargs:
        Keyword arguments to the function or class-method.

    :return:
        The result of calling the function or class-method.
    """

    # If the cache-file exists.
    if os.path.exists(cache_path):
        # Load the cached data from the file.
        with open(cache_path, mode='rb') as file:
            obj = pickle.load(file)

        print("- Data loaded from cache-file: " + cache_path)
    else:
        # The cache-file does not exist.

        # Call the function / class-method with the supplied arguments.
        obj = fn(*args, **kwargs)

        # Save the data to the cache-file.
        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)

        print("- Data saved to cache-file: " + cache_path)

    return obj

def convert_numpy2pickle(in_path, out_path):
    """
    Convert a numpy-file to pickle-file.
    """

    data = np.load(in_path)

    with open(out_path, mode='wb') as file:
        pickle.dump(data, file)