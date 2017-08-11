# -*- coding: utf-8 -*-
"""
All BB stores with last update in this year (7522 stores)
-----------------------------------------------------------------------------
"""
"""--------------------------------------------------------------------------
Load Data
-------------------------------------------------------------------------"""
#Load the CSV file into a RDD
mydir = "/Users/rachelguerin/Desktop/"
storeData = MySparkContext.textFile(mydir+"bbw_store.csv")
storeData.cache()
storeData.count()

#Remove the first line (contains headers)
firstLine = storeData.first()
dataLines = storeData.filter(lambda x: x != firstLine)
dataLines.count()

"""--------------------------------------------------------------------------
Cleanup Data
-------------------------------------------------------------------------"""

# Change labels to numeric ones and build a Row object

import math
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row

def transformToNumeric( inputStr) :
    #convert nulls to 0
    inputStr = inputStr.replace("NULL","0.0")
    
    attList=inputStr.split(",")

    #convert outcome to float    
    outcome = 1.0 if attList[2] == "2" else 0.0
    
    #Create a row with cleaned up and converted data
    values= Row(     OUTCOME= outcome,\
                    ID_STORE=float(attList[0]), \
                    BV_ELEGIBLE=float(attList[3]), \
                    BV_STATUS=float(attList[4]), \
                    CURRENCY=float(attList[5]), \
                    COUNTRY=float(attList[6]), \
                    LANGUAGE=float(attList[7]), \
                    REFERRER=float(attList[8]), \
                    PICKUP=float(attList[9]), \
                    COMPANY=float(attList[10]), \
                    DOMESTIC=float(attList[11]), \
                    INTERNATIONAL=float(attList[12]), \
                    HAS_PAYOUT=float(attList[13]), \
                    HAS_SOCIAL_NETWORK=float(attList[14]), \
                    HAS_LEGAL=float(attList[15]), \
                    HAS_SHARED=float(attList[16]), \
                    HAS_SHIPPING=float(attList[17]), \
                    HAS_PRODUCT=float(attList[18]), \
                    BANK_OUTSIDE_EU=float(attList[19]) , \
                    FACEBOOK_COUNT=float(attList[20]), \
                    BB_EXPRESS=float(attList[21]), \
                    BLOOMBEES_SHIPPING=float(attList[22]), \
                    COMPLIANCE=float(attList[23]), \
                    INSTAGRAM_COUNT=float(attList[24]), \
                    BP_TOT_PRD=float(attList[25]), \
                    BP_TOT_SALES=float(attList[26]), \
                    BP_TOT_TRANS=float(attList[27])
                    ) 
    return values
    
#Change to a Vector
##(top500,rest) = dataLines.randomSplit([0.5,0.5])
storeRows = dataLines.map(transformToNumeric)
storeRows.collect()[:15]

storeData2 = MySparkSession.createDataFrame(storeRows)

"""--------------------------------------------------------------------------
Perform Data Analytics
-------------------------------------------------------------------------"""
#See descriptive analytics.
storeData2.describe().show()

#Find correlation between predictors and target
for i in storeData2.columns:
    if not( isinstance(storeData2.select(i).take(1)[0][0], str)) :
        print( "Correlation between OUTCOME and ", i, \
            storeData2.stat.corr('OUTCOME',i))

"""--------------------------------------------------------------------------
Prepare data for ML
-------------------------------------------------------------------------"""
#Transform to a Data Frame for input to Machine Learing

def transformToLabeledPoint(row) :
    lp = ( row["OUTCOME"], \
            Vectors.dense([
                row["ID_STORE"], \
                row["BB_EXPRESS"], \
                row["COMPLIANCE"], \
                row["BLOOMBEES_SHIPPING"], \
                row["BV_ELEGIBLE"], \
                row["BV_STATUS"], \
                row["DOMESTIC"], \
                row["HAS_PAYOUT"], \
                row["HAS_PRODUCT"], \
                row["HAS_SHARED"], \
                row["HAS_SHIPPING"], \
                row["PICKUP"], \
                row["INTERNATIONAL"], \
                row["BP_TOT_PRD"],\
                row["BP_TOT_SALES"],\
                row["BP_TOT_TRANS"] \
        ]))
    return lp


storeLp = storeData2.rdd.map(transformToLabeledPoint)
storeLp.collect()
storeDF = MySparkSession.createDataFrame(storeLp,["label", "features"])
storeDF.select("label", "features").show(10)


"""--------------------------------------------------------------------------
Perform Machine Learning
-------------------------------------------------------------------------"""
#Perform PCA for feature extraction REMOVE so we can see which features count
"""
from pyspark.ml.feature import PCA
storePCA = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
pcaModel = storePCA.fit(storeDF)
pcaResult = pcaModel.transform(storeDF).select("label","pcaFeatures")
pcaResult.show(truncate=False)

#Indexing needed as pre-req for Decision Trees
from pyspark.ml.feature import StringIndexer
stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
si_model = stringIndexer.fit(pcaResult)
td = si_model.transform(pcaResult)
td.take(2)
"""
#Split into training and testing data
(trainingData, testData) = storeDF.randomSplit([0.7, 0.3])
trainingData.count()
testData.count()

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Create the model
rmClassifer = RandomForestClassifier(labelCol="label", \
                featuresCol="features", numTrees=100)
rmModel = rmClassifer.fit(trainingData)

#Predict on the test data
predictions = rmModel.transform(testData)
predictions.select("prediction","label","features").collect()
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="accuracy")
evaluator.evaluate(predictions)      

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()

#Balance data set 
from numpy.random import randint
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
 
RATIO_ADJUST = 2.0 ## ratio of pos to neg in the df_subsample
 
counts = trainingData.select('label').groupBy('label').count().collect()
higherBound = counts[0][1]
TRESHOLD_TO_FILTER = int(RATIO_ADJUST * float(counts[1][1]) / counts[0][1] * higherBound)
 
randGen = lambda x: randint(0, higherBound) if x == 0.0 else -1
 
udfRandGen = udf(randGen, IntegerType())
trainingData = trainingData.withColumn("randIndex", udfRandGen("label"))
df_subsample = trainingData.filter(trainingData['randIndex'] < TRESHOLD_TO_FILTER)
df_subsample = df_subsample.drop('randIndex')
 
print("Distribution of Pos and Neg cases of the down-sampled training data are: \n", df_subsample.groupBy("label").count().take(3))

#re-run ML over balanced data set
rmClassifer = RandomForestClassifier(labelCol="label", \
                featuresCol="features", numTrees=100)
rmModel = rmClassifer.fit(df_subsample)

#Predict on the test data
predictions = rmModel.transform(testData)
predictions.select("prediction","label","features").collect()
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="accuracy")
evaluator.evaluate(predictions)      

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()



#Predict on all data
predictions = rmModel.transform(storeDF)
predictions.select("prediction","label","features").collect()
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="accuracy")
evaluator.evaluate(predictions)      

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()

possiblePro = predictions.filter(predictions["prediction"] == 1.0)


def toCSVLine(data):
    line = data.split(",")
    r = line[1].replace("]","")
    return r

lines = possiblePro.rdd.map(toCSVLine)
lines.saveAsTextFile(mydir+'PROpredictions.csv')