from kNN import *
from numpy import array
import matplotlib
import matplotlib.pyplot as plot

datingMat,datingLabels,intDatingLabels = file2matrix('datingTestSet.txt')

fig = plot.figure()
ax = fig.add_subplot(111)
ax.scatter(datingMat[:,1], datingMat[:,2])
plot.show()

fig2 = plot.figure()
normDatingMat,ranges,minVal = autoNorm(datingMat)
ax = fig2.add_subplot(111)
ax.scatter(normDatingMat[:,1], normDatingMat[:,2])
plot.show()

fig3 = plot.figure()
ax = fig3.add_subplot(111)
ax.scatter(datingMat[:,1], datingMat[:,2],datingMat[:,3], 15.0*array(intDatingLabels), 15.0*array(intDatingLabels), 15.0*array(intDatingLabels))
plot.show()
