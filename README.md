# CC2020-Final-Project
Final Project: ASHRAE - Great Energy Predictor III

------------

## Introduction

According to NASA report, the carbon dioxide level was always below 300 parts per million for millennia. However, since 1950 the level has been rapidly increasing, and reaches approximately 415 parts per million nowadays. One of the main factors to the increase of carbon dioxide level is the energy consumption of industrial buildings. To address the increasing use of energy in industry, significant investments have been made to improve building efficiencies.

However, the challenge is whether the retrofits help to reduce costs and carbon dioxide emissions. A solution is to build a counterfactual machine learning model to forecast the amount of energy consumption for the original building using historic usage rates. The forecasts are compared to the actual energy consumption to calculate the savings after improvements.

## Dataset

The data for this project comes from the Kaggle competition ASHRAE - Great Energy Predictor III.

There are 3 datasets: the train.csv containing the meter reading, building.csv containing the information of buildings and weather_train.csv containing the weather status in different locations in a particular time. In summary, there are over 20 million observations and 16 variables. The target variable is the energy consumption recorded as meter reading. The predictors are the type of energy (i.e. electricity, chilled water, hot water and steam), the primary use of the building, the gross floor area, the time when the building was opened, the number of floors in the building and meteorological factors such as wind, cloud, temperature and pressure.



