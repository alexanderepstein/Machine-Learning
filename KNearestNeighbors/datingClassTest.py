from numpy import *
from kNN import *


if __name__ == "__main__":

    hoRatio = 0.10
    datingDataMat,datingLabels,datingNumLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVal = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorcount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print("The classifer came back with %s, which is %s \n" %(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorcount += 1.0
    print("The total error rate is: %f" % (errorcount/float(numTestVecs)))
