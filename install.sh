#!/usr/bin/env bash

# This script will get you up and running on a Google Compute Engine instance
# Ubuntu 16.04 (as many CPUs as you like)

# exit on failure
set -e

# install pip (not installed by default on GCE)
sudo apt install python-pip

# install virtualenv
sudo pip install virtualenv

# create and activate virtual environment
virtualenv venv
source venv/bin/activate

# install h5py
pip install h5py

# install tensorflow
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL

# install keras from source in the home directory
git clone https://github.com/fchollet/keras ~/keras/
python ~/keras/setup.py install --home ~/keras/
echo '{"epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow"}' > ~/.keras/keras.json

# alert user that we're done
echo ">==< Successfully installed dependencies >==<"

