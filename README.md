# Jetson Training

Training material created to cover the following topics:

* Introduction to the Jetson Nano hardware/software stacks
* Deep learning basics for computer vision
* TensorRT for model optimization
* DeepStream for creating video analytics pipelines

## Table of Contents

* [Jetson Nano Setup](#jetson-setup)
* [Content Walkthrough](#walkthrough)

## Jetson Nano Setup <a name="jetson-setup"></a>

There are two ways that you can setup everything necessary to run the contents of this lab.  The first is to simply download the Jetson image provided [IMAGE DOWNLOAD LINK COMING SOON]() which will include all of the necessary packages you will need for this material.  The image is based on the newest (at the time of creation) JetPack 4.4.

However, if you would like to build a fresh system yourself (or you already have your Jetson Nano setup and don't want to re-image), the following steps will allow you to install the correct packages on top of a JetPack install.

### JetPack 4.4

First, if you haven't already, grab the JetPack 4.4 SD card image for Jetson Nano from [here](https://developer.nvidia.com/embedded/jetpack).  Install using something like [BalenaEtcher](https://www.balena.io/etcher/) or equivalent software and follow the on-screen instructions to go through the initial setup of your Jetson Nano with JetPack 4.4.

### Package Installation

Once we have our OS setup, we want to make sure we have a few packages installed before going through the rest of the setup.

```bash
# Update the apt packages
$ sudo apt-get update

# Install pip3 for use with python3.6 (installed by default in JetPack 4.4)
$ sudo apt-get install python3-pip
```

#### TensorFlow Installation

Next, we want to install TensorFlow onto the Jetson Nano.  For this particular lab, we will be using TensorFlow 1.15, but there are `.whl` files for TensorFlow 2.x as well supported for the Jetson Nano.  Different versions and installation instructions can be found on the [Jetson Zoo](https://elinux.org/Jetson_Zoo).

```bash
$ sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
$ sudo pip3 install -U pip testresources setuptools
$ sudo pip3 install -U numpy==1.16.1 future==0.17.1 mock==3.0.5 h5py==2.9.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
$ sudo wget https://developer.download.nvidia.com/compute/redist/jp/v44/tensorflow/tensorflow-1.15.2+nv20.6-cp36-cp36m-linux_aarch64.whl
$ sudo pip3 install tensorflow-1.15.2+nv20.6-cp36-cp36m-linux_aarch64.whl
```

#### PyCUDA Installation

Next, we will use PyCUDA during our TensorRT section of our lab, so we need to install from source.

```bash
# Add CUDA to PATH/LD_LIBRARY_PATH/CPATH environment variables (add in ~./bashrc)
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
export CPATH=$CPATH:/usr/local/cuda-10.2/targets/aarch64-linux/include
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.2/targets/aarch64-linux/lib

# Source the changes to ~/.bashrc
$ source ~/.bashrc

# Now we want to build pycuda
$ wget https://files.pythonhosted.org/packages/5e/3f/5658c38579b41866ba21ee1b5020b8225cec86fe717e4b1c5c972de0a33c/pycuda-2019.1.2.tar.gz
$ tar xfz pycuda-2019.1.2.tar.gz
$ cd pycuda-2019.1.2
$ python3 configure.py --cuda-root=/usr/local/cuda
$ sudo make install

# Test to make sure PyCUDA is installed correctly
$ sudo pip3 install -U pytest
$ cd test
$ python3 test_driver.py
```

NOTE: There will be one test that fails on Jetson Nano.  This is OK as we are not using this portion in our testing or during the lab material.

#### JupyterLab Installation/Configuration

Last of that packages we need to install is JupyterLab.  We want to configure it as well so it's easier to use later.

```bash
# Install JupyterLab and other Python packages
$ sudo pip install -U pillow
$ sudo apt-get install libfreetype6-dev pkg-config
$ sudo pip3 install -U matplotlib
$ sudo pip3 install -U jupyter jupyterlab

# Generate a Jupyter Notebook config file
$ jupyter lab --generate-config
# Uncomment/change the following lines (# == line-number)
(204) c.NotebookApp.ip = '*'
(266) c.NotebookApp.notebook_dir = '/home/nvidia/jetson-training'
(272) c.NotebookApp.open_browser = False
(345) c.NotebookApp.token = 'nvidia'
```

#### DeepStream Installation

Now that we have all of our other packages installed, let's install the last SDK that we will be using, the [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk).

```bash
# Install DeepStream
# Download tar file for Jetson from here: https://developer.nvidia.com/deepstream-getting-started
$ cd ~/Downloads
$ sudo apt install libssl1.0.0 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4=2.11-1
$ sudo apt-get install librdkafka1=0.11.3-1build1

# Edit the /etc/apt/sources.list.d/nvidia-l4t-apt-source.list file to make sure the following are there:
#   deb https://repo.download.nvidia.com/jetson/common r32.4 main
#   deb https://repo.download.nvidia.com/jetson/t210 r32.4 main
$ sudo apt update
$ sudo apt install --reinstall nvidia-l4t-gstreamer
$ sudo tar -xvpf deepstream_sdk_v5.0.0_jetson.tbz2 -C /
$ cd /opt/nvidia/deepstream/deepstream-5.0
$ sudo ./install.sh sudo ldconfig

# Test out Deepstream to make sure it's working (NOTE: if you do the manual install you will not have source2.txt, just use the source4_...txt file to test)
$ deepstream-app -c /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/source2.txt
```

## Content Walkthrough <a name="walkthrough"></a>

The content being presented here is based on the examples which are present in the TensorFlow container on NVIDIA GPU Cloud ([NGC](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)).  You will notice there are alot of files providing functionality like an inference wrapper.  These code snippets are taken directly from the samples located in the container.

The content is split up into 3 sections:

* [Image Classification w/ TensorRT](1-ImageClassification-TRT.ipynb)
* [Object Detection w/ TensorRT](2-ObjectDetection-TRT.ipynb)
* [DeepStream](3-DeepStream.ipynb)

The first of the notebooks walks through creating a very simply model (in this case, a variant of the LeNet-5 model) for image classification.  We go through the process of creating this model from scratch using the TensorFlow/Keras API, downloading the MNIST dataset for training, and go through a few epochs of training.  Once we have our model, we can then go about optimizing it for deployment on the Jetson Nano.  In these notebooks, we will be using an intermediate file format called [UFF](https://en.wikipedia.org/wiki/Universal_File_Format) (note that there are others including [ONNX](https://onnx.ai/) which is very popular in the machine learning and deep learning world).  We walk through the process of converting our TensorFlow/Keras model into this intermediate file format (which will be more important in the second notebook when we have an "unsupported" layer in the network.  After we have our UFF model created, we then walk through the necessary processes to build and execute our TensorRT model (henceforth called an `engine`).

The second of the notebooks is an extension of the first where we use the same process, but take a much more complex model (in this case SSD) for object detection and convert it to a TensorRT engine.  We walk through the same process here of converting out TensorFlow frozen graph to a UFF model and then converting the UFF model to a TensorRT engine.  The main difference here is that we have a few layers which are not supported by the ONNX parser.  In this case, we have to add some "dummy nodes" in our model so that TensorRT knows how to handle them when creating the engine.

The last of the notebooks takes a step back and discusses a collection of applications called video analytics and introduces the NVIDIA DeepStream SDK.  We walk through how to use the SDK out of the box through modifying configuration files for your specific use case.  We then move on to a section illustrating how this can be done with the python API, building the entire pipeline from scratch.


## Licenses and Agreements
* [MIT License](LICENSE)
* [Individual Contributor License Agreement (CLA)](https://gist.github.com/alex3165/0d70734579a542ad34495d346b2df6a5)
