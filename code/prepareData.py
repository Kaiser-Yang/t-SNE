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
    if len(sys.argv) >= 2:
        n = int(sys.argv[1])
    print(f"set the number of samples be {n}.")
    if len(sys.argv) >= 3:
        epoch = int(sys.argv[2])
    print(f"set the number of epoch be {epoch}.")
    outputDimension = 2
    perp = 40
    dataLoader = loadData(n)
    item = enumerate(dataLoader)
    _, (img, label) = next(item)
    img = img.squeeze()
    x = np.zeros((n, img.shape[1] * img.shape[2]))
    for i in range(n):
        x[i] = img[i].flatten()
    pca = PCA(n_components = 30)
    newX = pca.fit_transform(x)
    dataFile = open("data/data.in", "w")
    labelFile = open("data/label.txt", "w")
    dataFile.writelines([str(newX.shape[0]) + '\n', str(newX.shape[1]) + '\n',
                         str(outputDimension) + '\n', str(perp) + '\n',
                         str(epoch) + '\n'])
    for i in range(n):
        xi = str(newX[i])
        xi = xi.replace('[', '')
        xi = xi.replace(']', '')
        dataFile.write(xi + '\n')
        labelFile.write(str(label[i].flatten().tolist()[0]) + "\n")
    dataFile.close()
    labelFile.close()