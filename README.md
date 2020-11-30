# CC2020-Final-Project
Final Project: ASHRAE - Great Energy Predictor III

------------

## Introduction

<div style="text-align: justify">
According to NASA report, the carbon dioxide level was always below 300 parts per million for millennia. However, since 1950 the level has been rapidly increasing, and reaches approximately 415 parts per million nowadays. One of the main factors to the increase of carbon dioxide level is the energy consumption of industrial buildings. To address the increasing use of energy in industry, significant investments have been made to improve building efficiencies.</div>

However, the challenge is whether the retrofits help to reduce costs and carbon dioxide emissions. A solution is to build a counterfactual machine learning model to forecast the amount of energy consumption for the original building using historic usage rates. The forecasts are compared to the actual energy consumption to calculate the savings after improvements.

## Dataset

The data for this project comes from the Kaggle competition ASHRAE - Great Energy Predictor III.

There are 3 datasets: the train.csv containing the meter reading, building.csv containing the information of buildings and weather_train.csv containing the weather status in different locations in a particular time. In summary, there are over 20 million observations and 16 variables. The target variable is the energy consumption recorded as meter reading. The predictors are the type of energy (i.e. electricity, chilled water, hot water and steam), the primary use of the building, the gross floor area, the time when the building was opened, the number of floors in the building and meteorological factors such as wind, cloud, temperature and pressure.

## Scope and Objectives

Due to the large sample size of the data, a virtual machine is required for data preprocessing and training and fine-tuning the models. In this project, we use AWS services to handle these tasks. S3 is used for storage of data and results. AWS Athena with interactive SQL queries is used for roughly analysis and joining the datasets. AWS Sagemaker provides jupyter notebook with preinstalled packages and libraries and association to github repository, which are helpful for feature engineering and building models. A private VPC network is assigned to AWS Sagemaker instance to ensured secure programming in the notebook.

The primary purpose of this project is to build the counterfactual model to forecast energy consume of original building from historic data and then compare it with actually energy consumption with the retrofit. The saving would help large scale investors and financial institutions clearly see the effective of the improvement and become more inclined to invest in this area to enable progress in building efficiencies. Moreover, this project is expected to provide a useful empirical experience relating to Cloud Computing using AWS services.
