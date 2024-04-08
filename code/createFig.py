#!/usr/bin/env python
import matplotlib.pyplot as plot
import imageio.v2 as imageio
import sys

if __name__ == "__main__":
    n = 6000
    epoch = 1000
    enableRandomWalk = 0
    argc = len(sys.argv)
    if argc >= 2:
        n = int(sys.argv[1])
    print(f"set the number of landmark samples be {n}.")
    if argc >= 3:
        epoch = int(sys.argv[2])
    print(f"set the number of epoch be {epoch}.")
    if argc >= 4:
        enableRandomWalk = int(sys.argv[3])
    print("creating figures...")
    labelSize = 10
    outputDimension = 2
    label = [0] * n
    labelFile = open("data/label.txt", "r")
    labelFileContent = list(map(int, labelFile.readlines()))
    yFile = open("data/data.out", "r")
    selectedID = list(map(int, yFile.readline().split()))
    for i in range(n):
        label[i] = labelFileContent[selectedID[i]]
    figNameAppendix = ""
    if enableRandomWalk:
        figNameAppendix = "_random_walk"
    for it in range(epoch):
        fig, ax = plot.subplots()
        result = [[[] for _ in range(outputDimension)] for _ in range(labelSize)]
        for i in range(n):
            y = list(map(float, yFile.readline().split()))
            for d in range(outputDimension):
                result[label[i]][d].append(y[d])
        for i in range(labelSize):
            ax.plot(result[i][0], result[i][1], "o")
        fig.savefig("fig/fig" + figNameAppendix + str(it) + ".png")
        plot.close()
    frameList = []
    for i in range(epoch):
        frameList.append(imageio.imread("fig/fig" + figNameAppendix + str(i) + ".png"))
    imageio.mimsave("result" + figNameAppendix + ".gif", frameList, "GIF", duration = 0.03)
    print("creating figures finished, check directories fig/ and ./")
    