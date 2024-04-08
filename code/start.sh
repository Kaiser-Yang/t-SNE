#!/bin/bash
n=6000
epoch=1000
enableRandomWalk=0
if [[ $# -ge 1 ]]; then
    n=$1
fi
totalSampleNum=$n
if [[ $# -ge 2 ]]; then
    epoch=$2
fi
if [[ $# -ge 3 ]]; then
    enableRandomWalk=$3
fi
if [[ $# -ge 4 ]]; then
    totalSampleNum=$4
fi
mkdir -p build
mkdir -p fig
mkdir -p data
rm -f fig/*
if [ -f "result.gif" ]; then
    rm result.gif
fi
if [ -f "result_random_walk.gif" ]; then
    rm result_random_walk.gif
fi
cd build || exit
cmake .. || exit
cmake --build . || exit
cd ..
python prepareData.py "$n" "$epoch" "$enableRandomWalk" "$totalSampleNum" || exit
build/tsne || exit
python createFig.py "$n" "$epoch" "$enableRandomWalk" || exit
