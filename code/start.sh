#!/bin/bash
mkdir -p build
mkdir -p fig
mkdir -p data
cd build
cmake ..
cmake --build .
cd ..
python prepareData.py
build/tsne
python createFig.py
