#!/usr/bin/bash

sudo apt update
sudo apt install nodejs
sudo apt install npm

cd frontend
npm install
npm audit fix
npm start
