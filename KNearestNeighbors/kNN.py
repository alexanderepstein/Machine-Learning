import operator
from numpy import *
from os import listdir

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
    distances = sqDistances**0.5
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

"""
This function normalizes all values in a dataSet o be between 0 and 1
"""

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals



def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percenTats = float(raw_input("Precentage of time spent playing video games: "))
    ffMiles = float(raw_input("Frequent flier miles earned per year: "))
    iceCream = float(raw_input("Liters of ice cream consumed per year: "))
    datingDataMat, datingLabels,datingNumLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percenTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingNumLabels,3)
    print("You will probably like this person in %s." % resultList[classifierResult-1])

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect



def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classiferResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("The classifer came back with: %d, the real answer is: %d %s" % (classiferResult,classNumStr,classiferResult==classNumStr))
        if (classiferResult != classNumStr):
            errorCount += 1.0
    print("\nThe total number of errors is: %d\n" % errorCount)
    print("The total error rate is: %f" % (errorCount/float(mTest)))
