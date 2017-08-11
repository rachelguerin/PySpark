# -*- coding: utf-8 -*-
"""
                    
Code Samples : Spark Machine Learning - Clustering

The input data contains samples of cars and technical / price 
information about them. The goal of this problem is to group 
these cars into 4 clusters based on their attributes

## Techniques Used

1. K-Means Clustering
2. Centering and Scaling

-----------------------------------------------------------------------------
"""
#import os
#os.chdir("C:/work")
#os.curdir

#Load the CSV file into a RDD
mydir = "/Users/rachelguerin/Desktop/"
storeData = MySparkContext.textFile(mydir+"bbw_store.csv")
storeData.cache()
storeData.take(2)


#Remove the first line (contains headers)
firstLine = storeData.first()
dataLines = storeData.filter(lambda x: x != firstLine)
dataLines.count()

from pyspark.sql import Row

import math
from pyspark.ml.linalg import Vectors

#Convert to Local Vector.
def transformToNumeric( inputStr) :
    # convert null values
    inputStr = inputStr.replace("NULL","0")
    
    attList=inputStr.split(",")
    
    
    #Filter out columns not wanted at this stage
    values= Row(FBFOLLOWERS= float(attList[0]), \
                     INSTFOLLOWERS=float(attList[1]),  \
                     TOTPRODUCTS=float(attList[2]),  \
                     TOTSALES=float(attList[3]),  \
                     TOTORDERS=float(attList[4]) \
                     )
    return values


storeMap = dataLines.map(transformToNumeric)
storeMap.persist()
storeMap.collect()

storeDf = MySparkSession.createDataFrame(storeMap)
storeDf.show()

#Centering and scaling. To perform this every value should be subtracted
#from that column's mean and divided by its Std. Deviation.

storeDf.describe().show()

summStats = storeDf.describe().toPandas()

meanValues = summStats.iloc[1,1:].values.tolist()
stdValues = summStats.iloc[2,1:].values.tolist()

#place the means and std.dev values in a broadcast variable
bcMeans = MySparkContext.broadcast(meanValues)
bcStdDev = MySparkContext.broadcast(stdValues)


def centerAndScale(inRow) :

    meanList = bcMeans.value
    stdList = bcStdDev.value

    outList = []
    for i in range(len(meanList)):
        outList.append( (float(inRow[i]) - float(meanList[i])) / float(stdList[i]) )
    return Vectors.dense(outList)
    

# autoDf is a dataframe, so a type of RDD
# to convert to a rdd and use its method we use dfname.rdd

csStore = storeDf.rdd.map(centerAndScale)
csStore.take(2)

#Create a Spark Data Frame from the rdd
storeRows = csStore.map( lambda f:Row(features=f))
storeDf = MySparkSession.createDataFrame(storeRows)
storeDf.show(3)

from pyspark.ml.clustering import KMeans
# 1.- define the algorithm to be used
kmeans = KMeans(k=2, seed=1)
# 2.- estimate the model (algorithm fit)
model = kmeans.fit(storeDf)
# 3.- use the fitted model to predict the class (creates a new DF with the predictions)
predictions = model.transform(storeDf)
predictions.show()

#Plot the results in a scatter plot

def unstripData(instr) :
    return ( instr["prediction"], instr["features"][0], \
        instr["features"][1],instr["features"][2],instr["features"][3], instr["features"][4])
    
unstripped = predictions.rdd.map(unstripData)
predList = unstripped.collect()

import pandas as pd
predPd = pd.DataFrame(predList)
predPd.columns = ['prediction', 'fbfollowers', 'instfollowers', 'totproducts', 'totsales', 'totorders']

import matplotlib.pylab as plt
#import seaborn 

plt.cla()
plt.scatter(predPd['fbfollowers'],predPd['totproducts'], c=predPd['prediction'])
plt.scatter(predPd['instfollowers'],predPd['totsales'], c=predPd['prediction'])
plt.scatter(predPd['totproducts'],predPd['totsales'], c=predPd['prediction'])
plt.scatter(predPd['totproducts'],predPd['totorders'], c=predPd['prediction'])
plt.scatter(predPd['fbfollowers'],predPd['totorders'], c=predPd['prediction'])
plt.scatter(predPd['instfollowers'],predPd['totorders'], c=predPd['prediction'])
plt.scatter(predPd['instfollowers'],predPd['totproducts'], c=predPd['prediction'])