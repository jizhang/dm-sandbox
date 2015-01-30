
# Machine Learning in Action

import numpy as np

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append([int(lineArr[2])])
    return np.array(dataMat), np.array(labelMat)

def sigmoid(inX):
    return 1.0 / (1.0 + np.exp(-inX))

def gradAscent(dataMat, labelMat):
    m, n = dataMat.shape
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMat.dot(weights))
        error = (labelMat - h)
        weights = weights + alpha * dataMat.T.dot(error)
    return weights

dataMat, labelMat = classLabels = loadDataSet()

wei = gradAscent(dataMat, labelMat)

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    n = dataMat.shape[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataMat[i, 1]); ycord1.append(dataMat[i, 2])
        else:
            xcord2.append(dataMat[i, 1]); ycord2.append(dataMat[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMat, labelMat):
    m, n = dataMat.shape
    alpha = 0.01
    weights = np.ones((n, 1))
    for i in range(m):
        h = sigmoid(np.sum(dataMat[i].dot(weights)))
        error = labelMat[i] - h
        weights = weights + alpha * dataMat[i].reshape((n, 1)) * error
    return weights

wei0 = stocGradAscent0(dataMat, labelMat)

def stocGradAscent1(dataMat, labelMat, numIter=150):
    m, n = dataMat.shape
    weights = np.ones((n, 1))
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(np.sum(dataMat[randIndex].dot(weights)))
            error = labelMat[randIndex] - h
            weights = weights + alpha * dataMat[randIndex].reshape((n, 1)) * error
            del(dataIndex[randIndex])
    return weights

wei1 = stocGradAscent1(dataMat, labelMat)
