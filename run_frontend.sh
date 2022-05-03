#!/usr/bin/bash

# please make sure that you are in image2geolocation folder

sudo apt update
sudo apt install nodejs
sudo apt install npm

cd frontend
npm install
npm audit fix
npm start
