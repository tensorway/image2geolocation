#!/usr/bin/bash

sudo apt update
sudo apt-get install wget

sudo wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh
bash Miniconda3-py39_4.11.0-Linux-x86_64.sh
rm Miniconda3-py39_4.11.0-Linux-x86_64.sh
