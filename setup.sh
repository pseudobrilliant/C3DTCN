#!/bin/bash

sudo apt-get install git python3-pip python3-dev build-essential swig python-wheel libcurl3-dev libfreetype6-dev libpng12-dev

sudo apt-get update

sudo pip3 install virtualenv

virtualenv --system-site-packages -p python3 ./CVEnv

source ./CVEnv/bin/activate

sudo pip3 install -r requirements.txt

#chmod a+x ixmas_setup.sh
#./ixmas_setup 
