#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ./Homework1
echo "Running particle filter now!"
python particle_filter.py --mask="./images/rutgers/WalkingMapMask.png" --json="./config.json"
echo "Particle filter done. Check out Homework1/animation/animation.gif to see the output"
deactivate