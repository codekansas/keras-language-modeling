#!/usr/bin/env bash

# This script will get you up and running on a Google Compute Engine instance
# Ubuntu 16.04 (as many CPUs as you like)

# exit on failure
# set -e

# make models directory
if [ ! -d "models/" ]; then
  mkdir models
fi

# install pip (not installed by default on GCE)
sudo apt install python-pip

# install virtualenv
sudo pip install virtualenv

# create and activate virtual environment
if [ ! -d "venv" ]; then
  virtualenv venv
fi
source venv/bin/activate

# install h5py
pip install h5py

# install blas/lapack
sudo apt install libblas-dev liblapack-dev libatlas-base-dev gfortran

# install tensorflow
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL

# install keras from source in the home directory
export KERAS_DIRECTORY=~/keras
if [ ! -d "${KERAS_DIRECTORY}" ]; then
  git clone https://github.com/fchollet/keras ${KERAS_DIRECTORY}
fi
cd $KERAS_DIRECTORY
python setup.py install
cd -
if [ ! -d ~/.keras ]; then
  mkdir ~/.keras
fi
echo '{"epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow"}' > ~/.keras/keras.json

# download insurance qa files
export INSURANCE_QA=~/insurance_qa
if [ ! -d $INSURANCE_QA ]; then
  git clone https://github.com/codekansas/insurance_qa_python $INSURANCE_QA
fi

# alert user that we're done
echo ">==< Successfully installed dependencies >==<"

