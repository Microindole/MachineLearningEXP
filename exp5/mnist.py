########################################################################
#
# Class for creating the MNIST data-set.
#
# This uses the same data-format as the TensorFlow tutorial,
# but wraps it in a Python class for easier use.
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

import numpy as np
import gzip
import os
import download
from dataset import one_hot_encoded

class MNIST:
    """
    The MNIST data-set for recognizing hand-written digits.
    """

    # The images are 28 pixels in each dimension.
    img_size = 28

    # The images are stored in one-dimensional arrays of this length.
    img_size_flat = img_size * img_size

    # Tuple with height and width of images used to reshape arrays.
    img_shape = (img_size, img_size)

    # Tuple with height, width and depth used to reshape arrays.
    # This is used for reshaping in Keras.
    img_shape_full = (img_size, img_size, 1)

    # Number of classes, one class for each of 10 digits.
    num_classes = 10

    # Number of colour channels for the images: 1 channel for gray-scale.
    num_channels = 1

    def __init__(self, data_dir="data/MNIST/"):
        """
        Load the MNIST data-set.
        Automatically downloads the files if they do not already exist locally.

        :param data_dir: Base-directory for downloading files.
        """

        # Copy args to self.
        self.data_dir = data_dir

        # The data-files are hosted at this URL.
        # Note: If Yann LeCun's server is down, you might need to change this mirror.
        # Common mirror: https://storage.googleapis.com/cvdf-datasets/mnist/
        self.base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

        # Filenames for the data-set.
        self.filename_x_train = "train-images-idx3-ubyte.gz"
        self.filename_y_train = "train-labels-idx1-ubyte.gz"
        self.filename_x_test = "t10k-images-idx3-ubyte.gz"
        self.filename_y_test = "t10k-labels-idx1-ubyte.gz"

        # Download the data-files if they don't already exist.
        self._download_data()

        # Load the training-set.
        self.x_train = self._load_images(filename=self.filename_x_train)
        self.y_train = self._load_cls(filename=self.filename_y_train)

        # Load the test-set.
        self.x_test = self._load_images(filename=self.filename_x_test)
        self.y_test = self._load_cls(filename=self.filename_y_test)

        # Helper for the test-set class numbers (alias for compatibility)
        self.y_test_cls = self.y_test

        # Prepare One-Hot Encoded labels if needed (optional for some tutorials)
        self.y_train_encoded = one_hot_encoded(class_numbers=self.y_train,
                                               num_classes=self.num_classes)
        self.y_test_encoded = one_hot_encoded(class_numbers=self.y_test,
                                              num_classes=self.num_classes)


    def _download_data(self):
        """
        Download all the data-files if they do not exist locally.
        """

        # List of filenames to download.
        filenames = [self.filename_x_train, self.filename_y_train,
                     self.filename_x_test, self.filename_y_test]

        for filename in filenames:
            # Create the full URL.
            url = self.base_url + filename

            # Download the file.
            download.maybe_download_and_extract(url=url, download_dir=self.data_dir)

    def _load_images(self, filename):
        """
        Load image-data from the given file.
        """

        # Full path for the file.
        file_path = os.path.join(self.data_dir, filename)

        print("Loading data: ", file_path)

        with gzip.open(file_path, 'rb') as f:
            # Read the data, skipping the first 16 bytes (header).
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        # Reshape the data to (num_images, img_size_flat).
        data = data.reshape(-1, self.img_size_flat)

        # Convert to floats between 0.0 and 1.0 (though original typically keeps uint8 until usage)
        # But Hvass-Labs tutorials usually return flattened arrays.
        return data

    def _load_cls(self, filename):
        """
        Load class-numbers from the given file.
        """

        # Full path for the file.
        file_path = os.path.join(self.data_dir, filename)

        print("Loading data: ", file_path)

        with gzip.open(file_path, 'rb') as f:
            # Read the data, skipping the first 8 bytes (header).
            data = np.frombuffer(f.read(), np.uint8, offset=8)

        return data