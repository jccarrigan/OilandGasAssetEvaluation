# Predicting Well Production

John Carrigan

--- 
Can oil production metrics be predicted using basic drilling, completion, and geologic data?

# Table of Contents

1. [Introduction](#introduction)
2. [Data](#data)
3. [Production Metric](#production-metric)
    * [Metric Determination](#metric-determination)
    * [Expanded Utility](#expanded-utility)
4. [Predictive Modeling](#predictive-modeling)
    * [Model Selection](#model-selection)
    * [Model Evaluation](#model-evaluation)
5. [Conclusions](#conclusions)
6. [Appendix](#appendix)
7. [References](#references)





## Introduction

 An undisclosed company is looking to purchase assets in an undisclosed location. To inform their investment an evaluation of the assets in question must be conducted. Forecasting production plays a large role in estimating an asset’s profitability. Traditionally this is done through an engineering approach that can be data, time,and labor intensive. Oil production is influenced by a variety of factors, from the inherent reservoir characteristics, to the specific method of drilling and completing each well. Acquiring the data to perform this analysis can also be prohibitively expensive.The objective of this project is to determine if this process can be replicated in a more efficient and less time and data intensive manner.
 
 ### Objectives
 
   * Create a model using basic drilling, completion, and geologic data to predict general oil and gas production
   * Determine which variables were most predictive of oil production volumes. 

---

## Data

The dataset included 13000 wells and 14 different files containing various geologic, drilling, and completion data, which is not made available.

### Methodology 

The data first had to be joined together using Pandas on API Number, a unique number given to all wells drilled domestically. A non-nominal amount of variables were neglected during this process, with the information deemed irrelevant, or too sparse to be of use. When completed, the final feature matrix included ~50 numerical variables and ~20 categorical values. Much of the dataset possessed missing values, with the range of missing data amounting to 5-75% missing values. Due to the skewed nature of many of the variables, the median of each feature was used to replace missing values.

At this point, multiple approaches were taken to determine the optimal feature matrix to train a model on. These include:
   * Numerical values only
   * Reducing dimensionality of categorical dummy variables using Principal Component Analysis (PCA)
   * Using 10 most important features from previous models (more detail below)
   
 **Principal Component Analysis**
 
 Give a brief technical explanation for how PCA works with graphs blah blah.  

## Production Metric

Three different metrics used to describe production over time: average, cumulative, and peak production. Wells must be drilled within the last 3 and 5 years respectively, have 2 and 4 years of production, and minimal shut in months

### Metric Determination

### Expanded Utility 

## Predictive Modeling

### Model Selection

A variety of regression models were tried, inluding Linear Regression, Random Forest, and multiple Gradient-Boosted models. Models were scored based on root mean-squared error. The final model selected was the Yandex algorithm CatBoosterRegressor. Once the model was chosen the hyperparameters were optimized using a grid search. 

### Model Evaluation

## Conclusions

Two key features for predicting future oil production: test volumes upon initial
production and frac volumes. Given the relative accuracy of the models, more investigation is necessary for these models to supplant traditional engineering techniques to evaluate a particular asset. This model may be useful in circumstances where time or data is limited.




## Appendix

## References

  * [Production Performance Metrics](http://www.verdazo.com/blog/what-production-performance-measure-should-i-use/)

* Tech Stack
   * Python
   * Numpy
   * Pandas
   * Sci-Kit Learn
   * Catboost
   * Seaborn
   * Matplotlib
