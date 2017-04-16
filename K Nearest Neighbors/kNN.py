import operator
from numpy import *


"""
This function creates a siimple dataset and corresponding list of classifcations
"""


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


"""
This function performs the classifcation on a an inX
that is the same amount of contains the same number of coluns are the dataSet
The labels are lined up ith the dataSet as this set of Data has already been classified
Because you always pass in a new data set to do classify the input no training occurs
to achieve classifcation
"""


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**.05
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename, "r")
    line = fr.readline()
    line = line.strip()
    listFromLine = line.split('\t')
    numberOfFeatures = len(listFromLine) - 1
    fr = open(filename, "r")
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, numberOfFeatures))
    numberClassLabels = []
    classLabelDict = {}
    classLabelVector = []
    fr = open(filename)
    dictionaryIndicie = 1
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:numberOfFeatures]
        classLabelVector.append(listFromLine[-1])
        if not classLabelDict.has_key(classLabelVector[index]):
            classLabelDict[classLabelVector[index]] = dictionaryIndicie
            dictionaryIndicie += 1
        numberClassLabels.append(classLabelDict.get(classLabelVector[index]))
        index += 1

    return returnMat, classLabelVector, numberClassLabels



def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
