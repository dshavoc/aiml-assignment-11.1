# Introduction

This is a homework assignment for an AI/ML class from UC Berkeley, 2025.  
Required Assignment 11.1

Jupyter Notebook: https://github.com/dshavoc/aiml-assignment-11.1/blob/master/assignment11_1.ipynb

# Overview

In this assignment I explore and analyze a provided data set on used cars (pared down from a dataset found on Kaggle) following a lite CRISP-DM process.

# Process

## Business Understanding

The client is a used-car lot. They need to know what factors drive price of a used car so they can set competitive prices and maximize profit.

The task in terms of data is to identify relationships between parameters and vehicle price using multiple data formulations and regression models. Each model will estimate the degree to which each feature contributes to the price. The features that contribute most positively or negatively to price will be of particular interest to the used car lot.

But first, the data must be cleaned; columns transformed and normalized as needed, and potentially combined.

On data reduncancy: there are several columns that carry repeated information, such as {'state' and 'region'}, {'manufacturer' and 'model'}. It could be argued that all the information contained in 'state' is represented by 'region' and so 'state' may be dropped. Likewise, 'make' could be dropped. Dropping the redundant columns would reduce the dimensionality of the data, but would discard some information that may be directly relevant to the client. It's possible the client cares about a trend by state or manufacturer, and those trends can be read directly from the model if the redundant columns are retained. That may find value if the number of columns (features) they add to the data is not too great a drawback.



## Data Understanding

The source data has many problems that must be addressed.

The original data has 426,880 samples (rows).

1. Columns with little missing data. Rows with missing data in these columns is dropped:
    - 'year' (<1%)
    - 'manufacturer' (4%)
    - 'model' (1%)
    - 'fuel' (1%)
    - 'odometer' (1%)
    - 'title_status' (2%)
    - 'transmission' (1%)

2. Drop columns that are not expected to bear significant predictive value:
    - 'VIN' - no predictive value
    - 'ID' - no predictive value
    - 'region' - 404 unique columns with limited predictive value

3. Fill missing values with 'NA' in columns that may bear significant predictive value:
    - 'drive' (missing 31%)
    - 'size' (missing 72%)
    - 'type' (missing 22%)
    - 'paint_color' (missing 31%)



### 
TODO: Wrangle some data.