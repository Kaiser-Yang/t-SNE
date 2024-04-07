#!/bin/bash
n=6000
epoch=1000
if [[ $# -ge 1 ]]; then
    n=$1
fi
if [[ $# -ge 2 ]]; then
    epoch=$2
fi
mkdir -p build
mkdir -p fig
mkdir -p data
rm -f fig/*
if [ -f "result.gif" ]; then
    rm result.gif
fi
cd build || exit
cmake ..
cmake --build .
cd ..
python prepareData.py "$n" "$epoch"
build/tsne
python createFig.py "$n" "$epoch"
