#!/usr/bin/env python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from sklearn.decomposition import PCA
import sys

def loadData(batchSize : int, path = "./data") -> DataLoader:
    trainData = datasets.MNIST(root = path, train = True, download = True, transform = transforms.ToTensor())
    testData = datasets.MNIST(root = path, train = False, download = True, transform = transforms.ToTensor())
    dataLoader = DataLoader(ConcatDataset([trainData, testData]), batch_size = batchSize, shuffle = True)
    return dataLoader

if __name__ == '__main__':
    n = 6000
    epoch = 1000
    argc = len(sys.argv)
    enableRandomWalk = 0
    if argc >= 2:
        n = int(sys.argv[1])
    print(f"set the number of landmark samples be {n}.")
    totalSampleNum = n
    if argc >= 3:
        epoch = int(sys.argv[2])
    print(f"set the number of epoch be {epoch}.")
    if argc >= 4:
        enableRandomWalk = int(sys.argv[3])
    if argc >= 5:
        totalSampleNum = int(sys.argv[4])
    if enableRandomWalk:
        print(f"enable random walk with totalSampleNum: {totalSampleNum}, landmarkSampleNum: {n}")
    print("preparing data...")
    outputDimension = 2
    perp = 40
    if enableRandomWalk:
        perp = 30
    dataLoader = loadData(totalSampleNum)
    item = enumerate(dataLoader)
    _, (img, label) = next(item)
    img = img.squeeze()
    x = np.zeros((totalSampleNum, img.shape[1] * img.shape[2]))
    for i in range(totalSampleNum):
        x[i] = img[i].flatten()
    pca = PCA(n_components = 30)
    newX = pca.fit_transform(x)
    dataFile = open("data/data.in", "w")
    labelFile = open("data/label.txt", "w")
    dataFile.writelines([str(n) + '\n', str(newX.shape[1]) + '\n',
                         str(outputDimension) + '\n', str(perp) + '\n',
                         str(epoch) + '\n', str(enableRandomWalk) + '\n',
                         str(totalSampleNum) + '\n'])
    for i in range(totalSampleNum):
        xi = str(newX[i])
        xi = xi.replace('[', '')
        xi = xi.replace(']', '')
        dataFile.write(xi + '\n')
        labelFile.write(str(label[i].flatten().tolist()[0]) + "\n")
    dataFile.close()
    labelFile.close()
    print("preparing data finished")
