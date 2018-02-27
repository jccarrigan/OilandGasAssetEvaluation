# Predicting Well Production Information

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

 The company in question is looking to purchase assets in an undisclosed location.To inform their investment an evaluation of the assets in question must be conducted. Forecasting production plays a large role in estimating an asset’s profitability. Traditionally this is done through an engineering approach that can be data, time,and labor intensive. Oil production is influenced by a variety of factors, from the inherent reservoir characteristics, to the specific method of drilling and completing each well. Acquiring the data to perform this analysis can also be prohibitively expensive.The objective of this project is to determine if this process can be replicated in a more efficient and less time and data intensive manner.

---

## Data

The dataset included 13000 wells and 14 different files containing various geologic, drilling, and completion data, which is not made available.  

**Target and Features**

  Features included various geologic, drilling, and completion data for each well. Categorical variables were converted to dummies using Pandas and then parsed down using Principal Component Analysis.
  
  Started with 14 different files, with information for ~13,000 unique wells and >150 unique features. Joined files together based upon API Number, parsed features down to ~40 features. Used PCA to reduce the dimensionality of dummy categorical variables to 20 components.
Created multiple production metrics from Time Series data. Train variety of models to predict production metrics, grid search on most effective models. Evaluate feature importance for most effective model
  
  Many different production metrics were created and used as targets in modeling. These included the average, cumulative, and peak production over various time frames. These time frames included the most recent two, three, and five years of production. 

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
