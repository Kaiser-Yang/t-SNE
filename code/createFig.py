#!/usr/bin/env python
import matplotlib.pyplot as plot
import imageio.v2 as imageio
import sys

if __name__ == "__main__":
    n = 6000
    epoch = 1000
    if len(sys.argv) >= 2:
        n = int(sys.argv[1])
    print(f"set the number of samples be {n}.")
    if len(sys.argv) >= 3:
        epoch = int(sys.argv[2])
    print(f"set the number of epoch be {epoch}.")
    labelSize = 10
    outputDimension = 2
    label = [0] * n
    labelFile = open("data/label.txt", "r")
    for i in range(n):
        label[i] = int(labelFile.readline())
    yFile = open("data/data.out", "r")
    for it in range(epoch):
        fig, ax = plot.subplots()
        result = [[[] for _ in range(outputDimension)] for _ in range(labelSize)]
        for i in range(n):
            y = list(map(float, yFile.readline().split()))
            for d in range(outputDimension):
                result[label[i]][d].append(y[d])
        for i in range(labelSize):
            ax.plot(result[i][0], result[i][1], "o")
        fig.savefig("fig/fig" + str(it) + ".png")
        plot.close()

    frameList = []
    for i in range(epoch):
        frameList.append(imageio.imread("fig/fig" + str(i) + ".png"))
    imageio.mimsave("result.gif", frameList, "GIF", duration = 0.03)
