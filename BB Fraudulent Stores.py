# -*- coding: utf-8 -*-
"""
   ML with pyspark (Spark+Python)

Problem Statement
*****************
The input data contains surveyed information about potential 
customers for a bank. The goal is to build a model that would 
predict if the prospect would become a customer if contacted by the marketing contact center

## Techniques Used

1. Random Forests
2. Training and Testing
3. Confusion Matrix
4. Indicator Variables
5. Variable Reduction
6. Principal Component Analysis

-----------------------------------------------------------------------------
"""
"""--------------------------------------------------------------------------
Load Data
-------------------------------------------------------------------------"""
#Load the CSV file into a RDD
mydir = "/Users/rachelguerin/Desktop/"
storeData = MySparkContext.textFile(mydir+"bbw_storeLEGAL.csv")
storeData.cache()
storeData.count()

#Remove the first line (contains headers)
firstLine = storeData.first()
dataLines = storeData.filter(lambda x: x != firstLine)

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
    outcome = 1.0 if attList[22] == "2" or attList[22] == "3"  else 0.0
    
    #Create a row with cleaned up and converted data
    values= Row(     OUTCOME= outcome,\
                    BV_ELEGIBLE=float(attList[2]), \
                    BV_STATUS=float(attList[3]), \
                    CURRENCY=float(attList[4]), \
                    COUNTRY=float(attList[5]), \
                    LANGUAGE=float(attList[6]), \
                    REFERRER=float(attList[7]), \
                    PICKUP=float(attList[8]), \
                    COMPANY=float(attList[9]), \
                    DOMESTIC=float(attList[10]), \
                    INTERNATIONAL=float(attList[11]), \
                    HAS_PAYOUT=float(attList[12]), \
                    HAS_SOCIAL_NETWORK=float(attList[13]), \
                    HAS_LEGAL=float(attList[14]), \
                    HAS_SHARED=float(attList[15]), \
                    HAS_SHIPPING=float(attList[16]), \
                    HAS_PRODUCT=float(attList[17]), \
                    BANK_OUTSIDE_EU=float(attList[18]) , \
                    FACEBOOK_COUNT=float(attList[19]), \
                    BB_EXPRESS=float(attList[20]), \
                    BLOOMBEES_SHIPPING=float(attList[21]), \
                    SUBSCRIPTION=float(attList[1]), \
                    INSTAGRAM_COUNT=float(attList[23]), \
                    BP_TOT_PRD=float(attList[24]), \
                    BP_TOT_SALES=float(attList[25]), \
                    BP_TOT_TRANS=float(attList[26]), \
                    ID_STORE=float(attList[28])
                    ) 
    return values
    
#Change to a Vector
#(top500,rest) = dataLines.randomSplit([0.5,0.5])
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
                row["BP_TOT_PRD"], \
                row["BP_TOT_SALES"], \
                row["BP_TOT_TRANS"], \
                row["BV_ELEGIBLE"], \
                row["BV_STATUS"], \
                row["HAS_PAYOUT"], \
                row["HAS_PRODUCT"], \
                row["HAS_SHARED"], \
                row["SUBSCRIPTION"] \
        ]))
    return lp


storeLp = storeData2.rdd.map(transformToLabeledPoint)
storeLp.collect()
storeDF = MySparkSession.createDataFrame(storeLp,["label", "features"])
storeDF.select("label", "features").show(10)


"""--------------------------------------------------------------------------
Perform Machine Learning
-------------------------------------------------------------------------"""
#Perform PCA for feature extraction
"""EXCLUDE FOR NOW
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
td.take(2)"""

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
predictions.select("prediction","indexed","label","pcaFeatures").collect()
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="indexed",metricName="accuracy")
evaluator.evaluate(predictions)      

#Draw a confusion matrix
predictions.groupBy("indexed","prediction").count().show()


#Balance data set - too many negatives
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

#get feature importance
d = rmModel.featureImportances

from matplotlib import pyplot as plt

x= [0,1,2,3,4,5,6,7]

y = [0.0489,0.0888,0.3526,0.2161,0.0873,0.1397,0.0601,0.0065]
N = len(y)

width = 0.35
plt.bar(x, y, width, color='r')


#Predict on the test data
predictions = rmModel.transform(testData)
predictions.select("prediction","indexed","label","pcaFeatures").collect()
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="indexed",metricName="accuracy")
evaluator.evaluate(predictions)      

#Draw a confusion matrix
predictions.groupBy("indexed","prediction").count().show()


# COMPARE TO LOGISTIC REGRESSION
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=10, regParam=0.5, elasticNetParam=0.8, \
       labelCol="indexed", featuresCol="pcaFeatures")
lrModel = lr.fit(trainingData)
#Predict on the test data
lrPredictions = lrModel.transform(testData)
lrPredictions.select("prediction","indexed","label","pcaFeatures").collect()
evaluator.evaluate(lrPredictions)


# COMPARE TO NEURAL NETWORK MULTILAYER PERCEPTRON
from pyspark.ml.classification import MultilayerPerceptronClassifier
layers = [3, 25, 25, 2]
# layers = [input_dim, internal layers, output_dim(number of classe) ]
nn = MultilayerPerceptronClassifier(maxIter=100, \
        layers=layers, \
    blockSize=128, seed=124, labelCol="indexed", \
    featuresCol="pcaFeatures")
nnModel = nn.fit(trainingData)
#Predict on the test data
nnPredictions = nnModel.transform(testData)
nnPredictions.select("prediction","indexed","label","pcaFeatures").collect()
evaluator.evaluate(nnPredictions)


"""--------------------------------------------------
Modify the code above to:
    - train a logistic regression with the original vars (5% significant p-value)
    - from the selected vars above, train 2 logistic models with regParam = [0.01 and 0.5]
    - train 2 random forest (number of trees = 10 and 100)
    - compare results
"""

#Create the model
rmClassifer10 = RandomForestClassifier(labelCol="indexed", \
                featuresCol="pcaFeatures", numTrees=10)
rmModel10 = rmClassifer10.fit(trainingData)

#Predict on the test data
predictionsTrain = rmModel10.transform(trainingData)
print( "Accuracy on training data:"+ str(evaluator.evaluate(predictionsTrain)) )
predictionsTest = rmModel10.transform(testData)
print( "Accuracy on test data:"+ str(evaluator.evaluate(predictionsTest)) )


#Create the model
rmClassifer100 = RandomForestClassifier(labelCol="indexed", \
                featuresCol="pcaFeatures", numTrees=100)
rmModel100 = rmClassifer100.fit(trainingData)

#Predict on the test data
predictionsTrain = rmModel100.transform(trainingData)
print( "Accuracy on training data:"+ str(evaluator.evaluate(predictionsTrain)) )
predictionsTest = rmModel100.transform(testData)
print( "Accuracy on test data:"+ str(evaluator.evaluate(predictionsTest)) )


lr01 = LogisticRegression(maxIter=10, regParam=0.01, elasticNetParam=0.8, \
       labelCol="indexed", featuresCol="pcaFeatures")
lrModel01 = lr01.fit(trainingData)

#Predict on the test data
predictionsTrain = lrModel01.transform(trainingData)
print( "Accuracy on training data:"+ str(evaluator.evaluate(predictionsTrain)) )
predictionsTest = lrModel01.transform(testData)
print( "Accuracy on test data:"+ str(evaluator.evaluate(predictionsTest)) )

lr5 = LogisticRegression(maxIter=10, regParam=0.5, elasticNetParam=0.8, \
       labelCol="indexed", featuresCol="pcaFeatures")
lrModel5 = lr5.fit(trainingData)

#Predict on the test data
predictionsTrain = lrModel5.transform(trainingData)
print( "Accuracy on training data:"+ str(evaluator.evaluate(predictionsTrain)) )
predictionsTest = lrModel5.transform(testData)
print( "Accuracy on test data:"+ str(evaluator.evaluate(predictionsTest)) )