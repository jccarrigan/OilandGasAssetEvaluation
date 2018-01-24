# Oil and Gas Asset Evaluation

Due to the sensitive nature of the data used in this project, many aspects of this project will be undisclosed. Discussion of my methodology in laying out and developing this particular project can be found below.



# Predicting Well Production Using Completion, Drilling, and Basic Reservoir Data

The dataset was 13000 wells and 14 different files containing various geologic, drilling, and completion data, which is not made available.  The final model used is the Yandex algorithm CatBoostRegressor. 

## Motivation

Oil production is influenced by a variety of factors, from the inherent reservoir characteristics, to the specific method of drilling and completing each well. Forecasting of production is of business interest so that operators can estimate the value of a particular well, as well as have an idea of how the well should perform into the future. In regards to evaluation, production forecasts allow operators to understand the profitability of their asset, and how much to value their assets at for stockholders or potential buyers. The company in question is looking to purchase assets in the undisclosed location. Having a better idea of future production in this area can provide clarity to where the company may want to invest their time and money in obtaining assets.


## Choice of Model

A variety of regression models were tried, inluding Linear Regression, Random Forest, and multiple Gradient-Boosted models. Models were scored based on root mean-squared error. The final model selected was the Yandex algorithm CatBoosterRegressor. Once the model was chosen the hyperparameters were optimized using a grid search. 


## Target and Features

  Features included various geologic, drilling, and completion data for each well. Categorical variables were converted to dummies using Pandas and then parsed down using Principal Component Analysis.
  
  Many different production metrics were created and used as targets in modeling. These included the average, cumulative, and peak production over various time frames. These time frames included the most recent two, three, and five years of production. 

## References

http://www.verdazo.com/blog/what-production-performance-measure-should-i-use/
