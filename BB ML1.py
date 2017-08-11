# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------
BB - predicted price of product?  

Spark Machine Learning - Linear Regression
 

Problem Statement
*****************
The input data set contains data about details of various car 
models. Based on the information provided, the goal is to come up 
with a model to predict Miles-per-gallon of a given model.

Techniques Used:

1. Linear Regression (multi-variate)
2. Data Imputation - replacing non-numeric data with numeric ones
3. Variable Reduction - picking up only relevant features

-----------------------------------------------------------------------------
"""

"""--------------------------------------------------------------------------
Load Data
-------------------------------------------------------------------------"""

#Load the CSV file into a RDD
mydir = "/Users/rachelguerin/Desktop/"
prodData = MySparkContext.textFile(mydir+"bbw_products.csv")
prodData.cache()
prodData.take(5)

#Remove the first line (contains headers)
firstLine = prodData.first()
dataLines = prodData.filter(lambda x: x != firstLine)
dataLines.count()

"""--------------------------------------------------------------------------
Cleanup Data
- input mising values
- covert to double format (from string)
-------------------------------------------------------------------------"""



from pyspark.sql import Row


#Function to cleanup Data
def CleanupData( inputStr) :
    attList=inputStr.split(",")
    
    if (attList[3]):
        for :
            numVariants++
        
            
    #Create a row with cleaned up and converted data
    values= Row(     PRICE=float(attList[0]),\
                     UNITS=float(attList[1]), \
                     PRICE_ORIG=float(attList[2]), 
                     VARIANTS=float(numVariants)) 
    return values

#Run map for cleanup
prodMap = dataLines.map(CleanupData)
prodMap.cache()
prodMap.take(5)
#Create a Data Frame with the data. 
prodDf = MySparkSession.createDataFrame(prodMap)
prodDf.describe().show()
prodDf.describe().toPandas()


"""--------------------------------------------------------------------------
Perform Data Analytics
two tasks:
1.- convert data to labeled points
2.- choose only the variables with high correlation (previous step)
-------------------------------------------------------------------------"""

#Find correlation between predictors and target
for i in prodDf.columns:
    if not( isinstance(prodDf.select(i).take(1)[0][0], str)) :
        print( "Correlation between PRICE and ", i, ' is ', prodDf.stat.corr('PRICE',i))

"""--------------------------------------------------------------------------
Prepare data for ML
-------------------------------------------------------------------------"""

# Transform to a Data Frame for input to Machine Learing
# Drop columns that are not required (low correlation)
# IMPORTANT: All machine learning algorithms in Spark need the data as a 
# DataFrame of labeledpoints

from pyspark.ml.linalg import Vectors

def transformToLabeledPoint(row) :
    lp = ( row["PRICE"], Vectors.dense([row["PRICE_ORIG"],\
                                     row["VARIANTS"],\
                        row["UNITS"]]))
    return lp
    
prodLp = prodMap.map(transformToLabeledPoint)

# now create a Labeledpoints DataFrame for calling machinelearning functions
prodDF = MySparkSession.createDataFrame(prodLp,["label", "features"])
prodDF.select("label","features").show(10)


"""--------------------------------------------------------------------------
Perform Machine Learning
-------------------------------------------------------------------------"""

#Split into training and testing data
(trainingData, testData) = prodDF.randomSplit([0.7, 0.3])
trainingData.count()
testData.count()

#Build the model on training data
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(maxIter=10)
lrModel = lr.fit(trainingData)

#Print the metrics
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))
print("standard errors: " + str(lrModel.summary.coefficientStandardErrors))
print("p-values: " + str(lrModel.summary.pValues))

#Predict on the test data
predictions = lrModel.transform(testData)
predictions.select("prediction","label","features").show()

# accuracy on test set: Find R2 for Linear Regression on test set
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(predictionCol="prediction",  labelCol="label", metricName="r2")
# other metrics (metricName) : rmse, mse, r2, mae 

evaluator.evaluate(predictions)


# COMPARE TO RANDOM FOREST REGRESSION
from pyspark.ml.regression import RandomForestRegressor
rf = RandomForestRegressor(numTrees=100)
rfModel = rf.fit(trainingData)
#Predict on the test data
rfPredictions = rfModel.transform(testData)
evaluator.evaluate(rfPredictions)


# GBT Regressor

from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(maxIter=30)
gbtModel = gbt.fit(trainingData)
#Predict on the test data
gbtPredictions = gbtModel.transform(testData)
evaluator.evaluate(gbtPredictions)


