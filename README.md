# Voyager 4

## How To Run

Linux and TensorFlow are required for running this implementation. We did it with Nvidia CUDA acceleration.

* TensorFlow 2.9
* Nvidia Cuda
* Conda
* NC (format for all datasets)
* Xarray (for extracting nc files in Py)
* Numpy
* Pandas
* Matplotlib
* Seaborn

### step 1: obtain data

This stack currently only supports NC data. The NC data are directly obtained from NOAA Portal or NEXT. They are to be placed directly in the w4 folder (indicating week 4). A future update will allow you to place it in a folder of your choosing.

### step 2: install tensorflow 2

Check the Google tutorials and carefully check your Nvidia CUDA versions to install TensorFlow 2 with GPU support.

### step 3: run

After extracting all the data, simply run `python3 ./ml.py`.

You are welcome to modify the individual machine learning models. This is a great way to get some experience with machine learning regression problems!

    See https://github.com/antonCPU/voyager-4-ml for the machine learning implementation.

The implementation can accommodate any number of data attributes and those designated as features or labels.
