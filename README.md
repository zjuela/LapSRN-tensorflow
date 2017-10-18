# Tensorflow implementation of the paper "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution" (CVPR 2017)

This is a Tensorflow implementation using TensorLayer.
Original paper and implementation using MatConNet can be found on their [project webpage](http://vllab1.ucmerced.edu/~wlai24/LapSRN/).

### Environment
The implementation is tested using python 3.6 and cuda 8.0.

### Download repository:

    $ git clone https://github.com/zjuela/LapSRN-tensorflow.git

### Train model
Specify dataset path in config.py file and run:

	$ python main.py

The pre-trained model is trained using [NTIRE 2017](http://www.vision.ee.ethz.ch/ntire17/) challenge dataset.

### Test
Run with your test image:

	$ python main.py -m test -f TESTIMAGE

Results can be find in folder ./samples/






