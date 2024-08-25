# Voyager 4

< This is a recently revived work in progress, and will be the monthly project for September. >

In this project, we present a solution to the problem of using anomalous and noisy "Level 1" raw data from the DSCOVR Faraday Cup to predict geomagnetic storms on Earth. Using this data, the corresponding "Level 2" data, and the accepted, "ground truth" Kp-index data starting from October 2015 when DSCOVR was launched, we created a machine learning algorithm that, after being trained on the ground truth Kp-index data, takes in anomalous Level 1 data and generates a synthetic set of Level 2 data that would not have been produced with the anomaly-ridden Level 1 data. Then, using this new and improved Level 2 data, we apply an algorithm to calculate the predicted Kp-index, which we display in a chart in 3-hour intervals on the website we created, similar to NOAA's "Planetary K-Index" page on their "Current Space Weather Conditions" website. Our website will host a feature allowing users to query the machine learning algorithm to output a Kp-index value based on the user's current time, which will allow the user to forecast how severe an incoming geomagnetic storm will be up to 2 hours in advance of its effects being felt on Earth. Naturally, there is great importance in being able to predict the strength of incoming geomagnetic storms, as it can provide valuable information such as how far south an aurora borealis will be visible and whether satellite communications and electrical grids will be disrupted. With future improvements, we believe our work will also strengthen our understanding of geomagnetic phenomena as a whole.

This application was originally conceived as a [NASA SpaceApps submission](https://www.spaceappschallenge.org/2023/find-a-team/voyager-2/?tab=project).

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

* See https://github.com/antonCPU/voyager-4-magnetic-alert (private) for the full stack implementation.

The implementation can accommodate any number of data attributes and those designated as features or labels.

## The Model

< Work in progress >
