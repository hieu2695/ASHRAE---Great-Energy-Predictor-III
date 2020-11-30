<h1 style = 'text-align: center;'>DATS 6450 Final Project</h1>
<h1 style = 'text-align: center;'>ASHRAE - Great Energy Predictor III</h1>
<h3 style = 'text-align: center;'>Authors: Tran Hieu Le, Voratham Tiabrat</h3>

<h1>Table of Contents<span class="tocSkip"></span></h1>

<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span>
    
<li><span><a href="#Dataset" data-toc-modified-id="Dataset-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Dataset</a></span>

<li><span><a href="#Scope-of-the-Project-and-Objectives" data-toc-modified-id="Scope-of-the-Project-and-Objectives-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Scope of the Project and Objectives</a></span>

<li><span><a href="#Data-Preprocessing" data-toc-modified-id="Data-Preprocessing-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Data Preprocessing</a></span><ul class="toc-item"><li><span><a href="#Import-Libraries" data-toc-modified-id="Import-Libraries-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Import Libraries</a></span></li><li><span><a href="#Load-data" data-toc-modified-id="Load-data-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Load data</a></span></li><li><span><a href="#Data-information" data-toc-modified-id="Data-information-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Data information</a></span></li><li><span><a href="#Getting-the-name-of-target-variable" data-toc-modified-id="Getting-the-name-of-target-variable-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Getting the name of target variable</a></span></li><li><span><a href="#Handling-rows-with-missing-data-for-target-variable" data-toc-modified-id="Handling-rows-with-missing-data-for-target-variable-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Handling rows with missing data for target variable</a></span></li><li><span><a href="#Handling-rows-with-abnormal-data-for-target-variable" data-toc-modified-id="Handling-rows-with-abnormal-data-for-target-variable-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Handling rows with abnormal data for target variable</a></span></li><li><span><a href="#Dividing-the-training-data-into-training-and-validation" data-toc-modified-id="Dividing-the-training-data-into-training-and-validation-4.7"><span class="toc-item-num">4.7&nbsp;&nbsp;</span>Dividing the training data into training and validation</a></span></li><li><span><a href="#Handling-date-time-variables" data-toc-modified-id="Handling-date-time-variables-4.8"><span class="toc-item-num">4.8&nbsp;&nbsp;</span>Handling date time variables</a></span></li><li><span><a href="#Handling-missing-data" data-toc-modified-id="Handling-missing-data-4.9"><span class="toc-item-num">4.9&nbsp;&nbsp;</span>Handling missing data</a></span><ul class="toc-item"><li><span><a href="#Identifying-missing-values" data-toc-modified-id="Identifying-missing-values-4.9.1"><span class="toc-item-num">4.9.1&nbsp;&nbsp;</span>Identifying missing values</a></span></li><li><span><a href="#Dropping-columns-with-high-proportion-of-missing-values" data-toc-modified-id="Dropping-columns-with-high-proportion-of-missing-values-4.9.2"><span class="toc-item-num">4.9.2&nbsp;&nbsp;</span>Dropping columns with high proportion of missing values</a></span></li><li><span><a href="#Removing-rows-of-variables-with-small-proportion-of-missing-values" data-toc-modified-id="Removing-rows-of-variables-with-small-proportion-of-missing-values-4.9.3"><span class="toc-item-num">4.9.3&nbsp;&nbsp;</span>Removing rows of variables with small proportion of missing values</a></span></li><li><span><a href="#Imputing-missing-values" data-toc-modified-id="Imputing-missing-values-4.9.4"><span class="toc-item-num">4.9.4&nbsp;&nbsp;</span>Imputing missing values</a></span></li></ul></li><li><span><a href="#Changing-data-type" data-toc-modified-id="Changing-data-type-4.10"><span class="toc-item-num">4.10&nbsp;&nbsp;</span>Changing data type</a></span></li><li><span><a href="#Encoding-the-categorical-variables" data-toc-modified-id="Encoding-the-categorical-variables-4.11"><span class="toc-item-num">4.11&nbsp;&nbsp;</span>Encoding the categorical variables</a></span></li><ul class="toc-item"><li><span><a href="#Identifying-missing-values" data-toc-modified-id="Identifying-missing-values-4.11.1"><span class="toc-item-num">4.11.1&nbsp;&nbsp;</span>Identifying missing values</a></span></li><li><span><a href="#One-hot-encoding-the-categorical-variables" data-toc-modified-id="One-hot-encoding-the-categorical-variables-4.11.2"><span class="toc-item-num">4.11.2&nbsp;&nbsp;</span>One-hot encoding the categorical variables</a></span></li><li><span><a href="#Seperating-training-and-validation-sets" data-toc-modified-id="Seperating-training-and-validation-sets-4.11.3"><span class="toc-item-num">4.11.3&nbsp;&nbsp;</span>Seperating training and validation sets</a></span></li></ul><li><span><a href="#Scaling-data" data-toc-modified-id="Scaling-data-4.12"><span class="toc-item-num">4.12&nbsp;&nbsp;</span>Scaling data</a></span></li><ul class="toc-item"><li><span><a href="#Transforming-target-variable-using-Natural-logarithm" data-toc-modified-id="Transforming-target-variable-using-Natural-logarithm-4.12.1"><span class="toc-item-num">4.12.1&nbsp;&nbsp;</span>Transforming target variable using Natural logarithm</a></span></li><li><span><a href="#Standardizing-data" data-toc-modified-id="Standardizing-data-4.12.2"><span class="toc-item-num">4.12.2&nbsp;&nbsp;</span>Standardizing data</a></span></li></ul><li><span><a href="#Getting-feature-matrix-and-target-vector" data-toc-modified-id="Getting-feature-matrix-and-target-vector-4.13"><span class="toc-item-num">4.13&nbsp;&nbsp;</span>Getting feature matrix and target vector</a></span></li></ul></li>

<li><span><a href="#Modeling" data-toc-modified-id="Modeling-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Modeling</a></span><ul class="toc-item"><li><span><a href="#Create-a-dictionary-of-the-models" data-toc-modified-id="Create-a-dictionary-of-the-models-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Create a dictionary of the models</a></span></li><li><span><a href="#Create-a-dictionary-of-the-pipelines" data-toc-modified-id="Create-a-dictionary-of-the-pipelines-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Create a dictionary of the pipelines</a></span></li><li><span><a href="#Hyperparameter-tuning-and-model-selection" data-toc-modified-id="Hyperparameter-tuning-and-model-selection-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Hyperparameter tuning and model selection</a></span><ul class="toc-item"><li><span><a href="#Getting-the-predefined-split-cross-validator" data-toc-modified-id="Getting-the-predefined-split-cross-validator-5.3.1"><span class="toc-item-num">5.3.1&nbsp;&nbsp;</span>Getting the predefined split cross-validator</a></span></li><li><span><a href="#Creating-the-dictionary-of-the-parameter-grids" data-toc-modified-id="Creating-the-dictionary-of-the-parameter-grids-5.3.2"><span class="toc-item-num">5.3.2&nbsp;&nbsp;</span>Creating the dictionary of the parameter grids</a></span></li><li><span><a href="#The-parameter-grid-for-LinearRegression" data-toc-modified-id="The-parameter-grid-for-LinearRegression-5.3.3"><span class="toc-item-num">5.3.3&nbsp;&nbsp;</span>The parameter grid for LinearRegression</a></span></li><li><span><a href="#The-parameter-grid-for-LightGBM" data-toc-modified-id="The-parameter-grid-for-LightGBM-5.3.4"><span class="toc-item-num">5.3.4&nbsp;&nbsp;</span>The parameter grid for LightGBM</a></span></li><li><span><a href="#The-parameter-grid-for-MLPRegressor" data-toc-modified-id="The-parameter-grid-for-MLPRegressor-5.3.5"><span class="toc-item-num">5.3.5&nbsp;&nbsp;</span>The parameter grid for MLPRegressor</a></span></li><li><span><a href="#Creating-the-directory-for-GridSearchCV-results" data-toc-modified-id="Creating-the-directory-for-GridSearchCV-results-5.3.6"><span class="toc-item-num">5.3.6&nbsp;&nbsp;</span>Creating the directory for GridSearchCV results</a></span></li></ul></li><li><span><a href="#GridSearchCV-Results" data-toc-modified-id="GridSearchCV-Results-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>GridSearchCV Results</a></span></li><li><span><a href="#Feature-Importance" data-toc-modified-id="Feature-Importance-5.5"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>Feature Importance</a></span></li></ul></li>
    
<li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Conclusion</a></span></li>
</div>


# Introduction

According to [NASA report](https://climate.nasa.gov/evidence/), the carbon dioxide level was always below 300 parts per million for millennia. 

<div style="text-align: justify"><br/> However, since 1950 the level has been rapidly increasing, and reaches approximately 415 parts per million nowadays. One of the main factors to the increase of carbon dioxide level is the energy consumption of industrial buildings. To address the increasing use of energy in industry, significant investments have been made to improve building efficiencies.</div>


<div style="text-align: justify"> <br/> However, the challenge is whether the retrofits help to reduce costs and carbon dioxide emissions. A solution is to build a counterfactual machine learning model to forecast the amount of energy consumption for the original building using historic usage rates. The forecasts are compared to the actual energy consumption to calculate the savings after improvements.</div>


# Dataset

The data for this project comes from the Kaggle competition [ASHRAE - Great Energy Predictor III](https://www.kaggle.com/c/ashrae-energy-prediction/data). 

<div style="text-align: justify">
    <br/>
There are 3 datasets: the train.csv containing the meter reading, building.csv containing the information of buildings and weather_train.csv containing the weather status in different locations in a particular time. In summary, there are over 20 million observations and 16 variables. The target variable is the energy consumption recorded as meter reading. The predictors are the type of energy (i.e. electricity, chilled water, hot water and steam), the primary use of the building, the gross floor area, the time when the building was opened, the number of floors in the building and meteorological factors such as wind, cloud, temperature and pressure.</div>


# Scope of the Project and Objectives

<div style="text-align: justify"> Due to the large sample size of the data, a virtual machine is required for data preprocessing and training and fine-tuning the models. In this project, we use AWS services to handle these tasks. S3 is used for storage of data and results. AWS Athena with interactive SQL queries is used for roughly analysis and joining the datasets. AWS Sagemaker provides jupyter notebook with preinstalled packages and libraries and association to github repository, which are helpful for feature engineering and building models. A private VPC network is assigned to AWS Sagemaker instance to ensured secure programming in the notebook.</div>

<div style="text-align: justify"><br/> The primary purpose of this project is to build the counterfactual model to forecast energy consume of original building from historic data and then compare it with actually energy consumption with the retrofit. The saving would help large scale investors and financial institutions clearly see the effective of the improvement and become more inclined to invest in this area to enable progress in building efficiencies. Moreover, this project is expected to provide a useful empirical experience relating to Cloud Computing using AWS services.</div>

# Data Preprocessing

## Import Libraries


```python
import boto3
import pandas as pd
import numpy as np
import sagemaker
```


```python
import matplotlib.pyplot as plt
%matplotlib inline 

# Set matplotlib sizes
plt.rc('font', size=10)
plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)
plt.rc('figure', titlesize=20)
```


```python
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')
```

## Load data
In this part, we will define the S3 bucket that is used for this project and load the dataset preprocessed by AWS Athena.


```python
# Get SageMaker session & default S3 bucket
role = sagemaker.get_execution_role()
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name
s3 = sagemaker_session.boto_session.resource('s3')
bucket='dataset.finalproject.cc2020'
data_path = 'fulldata/energy_dataset.csv'
data_location = 's3://{}/{}'.format(bucket, data_path)

df_raw = pd.read_csv(data_location) # load data from S3 bucket
```


```python
df_train = df_raw.copy(deep=True) # make a copy of the raw data
```


```python
df_train.head() # print first 5 rows of df_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meter</th>
      <th>meter_reading</th>
      <th>building_id</th>
      <th>primary_use</th>
      <th>square_feet</th>
      <th>year_built</th>
      <th>floor_count</th>
      <th>site_id</th>
      <th>timestamp</th>
      <th>air_temperature</th>
      <th>cloud_coverage</th>
      <th>dew_temperature</th>
      <th>precip_depth_1_hr</th>
      <th>sea_level_pressure</th>
      <th>wind_direction</th>
      <th>wind_speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>Education</td>
      <td>7432.0</td>
      <td>2008.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2016-05-11 06:00:00.000</td>
      <td>21.1</td>
      <td>8.0</td>
      <td>18.3</td>
      <td>0.0</td>
      <td>1020.4</td>
      <td>150.0</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.00</td>
      <td>7.0</td>
      <td>Education</td>
      <td>121074.0</td>
      <td>1989.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2016-05-11 06:00:00.000</td>
      <td>21.1</td>
      <td>8.0</td>
      <td>18.3</td>
      <td>0.0</td>
      <td>1020.4</td>
      <td>150.0</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2785.88</td>
      <td>7.0</td>
      <td>Education</td>
      <td>121074.0</td>
      <td>1989.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2016-05-11 06:00:00.000</td>
      <td>21.1</td>
      <td>8.0</td>
      <td>18.3</td>
      <td>0.0</td>
      <td>1020.4</td>
      <td>150.0</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.00</td>
      <td>11.0</td>
      <td>Education</td>
      <td>49073.0</td>
      <td>1968.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2016-05-11 06:00:00.000</td>
      <td>21.1</td>
      <td>8.0</td>
      <td>18.3</td>
      <td>0.0</td>
      <td>1020.4</td>
      <td>150.0</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.00</td>
      <td>14.0</td>
      <td>Education</td>
      <td>86250.0</td>
      <td>2013.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2016-05-11 06:00:00.000</td>
      <td>21.1</td>
      <td>8.0</td>
      <td>18.3</td>
      <td>0.0</td>
      <td>1020.4</td>
      <td>150.0</td>
      <td>2.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the dimension of df_train
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># rows</th>
      <th># columns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20216101</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>



## Data information


```python
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20216101 entries, 0 to 20216100
    Data columns (total 16 columns):
     #   Column              Dtype  
    ---  ------              -----  
     0   meter               float64
     1   meter_reading       float64
     2   building_id         float64
     3   primary_use         object 
     4   square_feet         float64
     5   year_built          float64
     6   floor_count         float64
     7   site_id             float64
     8   timestamp           object 
     9   air_temperature     float64
     10  cloud_coverage      float64
     11  dew_temperature     float64
     12  precip_depth_1_hr   float64
     13  sea_level_pressure  float64
     14  wind_direction      float64
     15  wind_speed          float64
    dtypes: float64(14), object(2)
    memory usage: 2.4+ GB


The meter column indicates the energy types which are categorical variable. Therefore, we need to convert the column into object type.


```python
# change meter dtypes from float64 to object
df_train['meter'] = df_train['meter'].astype('object') 
```


```python
df_train.info() # rechecking dtypes
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20216101 entries, 0 to 20216100
    Data columns (total 16 columns):
     #   Column              Dtype  
    ---  ------              -----  
     0   meter               object 
     1   meter_reading       float64
     2   building_id         float64
     3   primary_use         object 
     4   square_feet         float64
     5   year_built          float64
     6   floor_count         float64
     7   site_id             float64
     8   timestamp           object 
     9   air_temperature     float64
     10  cloud_coverage      float64
     11  dew_temperature     float64
     12  precip_depth_1_hr   float64
     13  sea_level_pressure  float64
     14  wind_direction      float64
     15  wind_speed          float64
    dtypes: float64(13), object(3)
    memory usage: 2.4+ GB


## Getting the name of target variable


```python
target = "meter_reading" # the prediction target
```

## Handling rows with missing data for target variable


```python
df_train[target].isnull().values.any()
```




    True




```python
df_train[df_train[target].isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meter</th>
      <th>meter_reading</th>
      <th>building_id</th>
      <th>primary_use</th>
      <th>square_feet</th>
      <th>year_built</th>
      <th>floor_count</th>
      <th>site_id</th>
      <th>timestamp</th>
      <th>air_temperature</th>
      <th>cloud_coverage</th>
      <th>dew_temperature</th>
      <th>precip_depth_1_hr</th>
      <th>sea_level_pressure</th>
      <th>wind_direction</th>
      <th>wind_speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10414</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train.dropna(subset = [target], inplace=True) # drop the nan row
```

## Handling rows with abnormal data for target variable

According to [this discussion](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/113054), the records for electricity consumption for buildings of site_id 0 are abnormal until May 20. Therefore, we should remove these values to improve the prediction.


```python
df_train = df_train.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
```


```python
# Print the dimension of df_train
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># rows</th>
      <th># columns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19869988</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>



## Dividing the training data into training and validation


```python
from sklearn.model_selection import train_test_split

# Divide the training data into training (80%) and validation (20%)
df_train, df_valid = train_test_split(df_train, train_size=0.8, random_state=42)

# Reset the index
df_train, df_valid = df_train.reset_index(drop=True), df_valid.reset_index(drop=True)
```


```python
df_train.head() # print first 5 rows of df_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meter</th>
      <th>meter_reading</th>
      <th>building_id</th>
      <th>primary_use</th>
      <th>square_feet</th>
      <th>year_built</th>
      <th>floor_count</th>
      <th>site_id</th>
      <th>timestamp</th>
      <th>air_temperature</th>
      <th>cloud_coverage</th>
      <th>dew_temperature</th>
      <th>precip_depth_1_hr</th>
      <th>sea_level_pressure</th>
      <th>wind_direction</th>
      <th>wind_speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>64.3250</td>
      <td>1403.0</td>
      <td>Lodging/residential</td>
      <td>78438.0</td>
      <td>2004.0</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>2016-01-27 06:00:00.000</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>-4.4</td>
      <td>NaN</td>
      <td>1015.5</td>
      <td>280.0</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>15.6250</td>
      <td>152.0</td>
      <td>Office</td>
      <td>10301.0</td>
      <td>1970.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2016-06-29 11:00:00.000</td>
      <td>15.3</td>
      <td>NaN</td>
      <td>12.2</td>
      <td>NaN</td>
      <td>1009.1</td>
      <td>210.0</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>83.3333</td>
      <td>850.0</td>
      <td>Public services</td>
      <td>28590.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>2016-04-09 17:00:00.000</td>
      <td>21.7</td>
      <td>2.0</td>
      <td>1.1</td>
      <td>0.0</td>
      <td>1020.2</td>
      <td>350.0</td>
      <td>4.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>600.3800</td>
      <td>79.0</td>
      <td>Office</td>
      <td>36240.0</td>
      <td>2010.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2016-11-30 13:00:00.000</td>
      <td>20.6</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>1015.7</td>
      <td>150.0</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>134.6130</td>
      <td>1090.0</td>
      <td>Office</td>
      <td>305047.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>2016-05-24 07:00:00.000</td>
      <td>19.4</td>
      <td>NaN</td>
      <td>16.1</td>
      <td>-1.0</td>
      <td>1008.5</td>
      <td>190.0</td>
      <td>2.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the dimension of df_train
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># rows</th>
      <th># columns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15895990</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_valid.head() # print first 5 rows of df_valid
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meter</th>
      <th>meter_reading</th>
      <th>building_id</th>
      <th>primary_use</th>
      <th>square_feet</th>
      <th>year_built</th>
      <th>floor_count</th>
      <th>site_id</th>
      <th>timestamp</th>
      <th>air_temperature</th>
      <th>cloud_coverage</th>
      <th>dew_temperature</th>
      <th>precip_depth_1_hr</th>
      <th>sea_level_pressure</th>
      <th>wind_direction</th>
      <th>wind_speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>692.220</td>
      <td>376.0</td>
      <td>Office</td>
      <td>585955.0</td>
      <td>1942.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>2016-11-04 13:00:00.000</td>
      <td>13.9</td>
      <td>2.0</td>
      <td>7.2</td>
      <td>0.0</td>
      <td>1022.8</td>
      <td>350.0</td>
      <td>6.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>475.000</td>
      <td>883.0</td>
      <td>Education</td>
      <td>399331.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>2016-07-09 20:00:00.000</td>
      <td>36.7</td>
      <td>2.0</td>
      <td>22.2</td>
      <td>0.0</td>
      <td>1014.8</td>
      <td>NaN</td>
      <td>2.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>183.404</td>
      <td>237.0</td>
      <td>Public services</td>
      <td>101262.0</td>
      <td>1982.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2016-09-04 05:00:00.000</td>
      <td>33.3</td>
      <td>0.0</td>
      <td>2.8</td>
      <td>0.0</td>
      <td>1002.4</td>
      <td>240.0</td>
      <td>4.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.000</td>
      <td>176.0</td>
      <td>Education</td>
      <td>62238.0</td>
      <td>1970.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2016-02-20 23:00:00.000</td>
      <td>29.4</td>
      <td>4.0</td>
      <td>-5.6</td>
      <td>0.0</td>
      <td>1012.1</td>
      <td>290.0</td>
      <td>4.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>171.597</td>
      <td>755.0</td>
      <td>Office</td>
      <td>42129.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>2016-08-18 14:00:00.000</td>
      <td>27.8</td>
      <td>0.0</td>
      <td>21.7</td>
      <td>0.0</td>
      <td>1018.2</td>
      <td>NaN</td>
      <td>2.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the dimension of df_valid
pd.DataFrame([[df_valid.shape[0], df_valid.shape[1]]], columns=['# rows', '# columns'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># rows</th>
      <th># columns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3973998</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>



## Handling date time variables

The "timestamp" column is recorded as object type combining year, month, day and hour. We need to separate the column into 4 specific columns including year, month, day and hour.


```python
import datetime

def datetime_transformer(df, datetime_vars):
    """
    The datetime transformer

    Parameters
    ----------
    df : dataframe
    datetime_vars : the datetime variables
    
    Returns
    ----------
    The dataframe where datetime_vars are transformed into the following 3 datetime types:
    year, month, day
    
    """
    # The dictionary with key as datetime type and value as datetime type operator
    dict_ = {'year'   : lambda x : x.dt.year,
             'month'  : lambda x : x.dt.month,
             'day'    : lambda x : x.dt.day,
             'hour'   : lambda x : x.dt.hour}
    
    # make a copy of df that will contain 3 datetime variables
    df_datetime = df.copy(deep=True)
    
    # for each column that is datetime variable
    for var in datetime_vars:
        # cast the integer data to datetime 
        df_datetime[var] = df_datetime[var].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d %H:%M:%S.%f'))
        
        
        # for each item (datetime_type and datetime_type_operator) in dict_
        for datetime_type, datetime_type_operator in dict_.items():
            # add a new variable to df_datetime where:
            # the variable's name is var + '_' + datetime_type
            # the variable's values are the ones obtained by datetime_type_operator
            df_datetime[var + '_' + datetime_type] = datetime_type_operator(df_datetime[var])
            
    # remove datetime_vars from df_datetime
    df_datetime = df_datetime.drop(columns=datetime_vars)
                
    return df_datetime
```


```python
# call datetime_transformer on df_train
df_train = datetime_transformer(df_train, ['timestamp'])

# print the first 5 rows of df_train
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meter</th>
      <th>meter_reading</th>
      <th>building_id</th>
      <th>primary_use</th>
      <th>square_feet</th>
      <th>year_built</th>
      <th>floor_count</th>
      <th>site_id</th>
      <th>air_temperature</th>
      <th>cloud_coverage</th>
      <th>dew_temperature</th>
      <th>precip_depth_1_hr</th>
      <th>sea_level_pressure</th>
      <th>wind_direction</th>
      <th>wind_speed</th>
      <th>timestamp_year</th>
      <th>timestamp_month</th>
      <th>timestamp_day</th>
      <th>timestamp_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>64.3250</td>
      <td>1403.0</td>
      <td>Lodging/residential</td>
      <td>78438.0</td>
      <td>2004.0</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>-4.4</td>
      <td>NaN</td>
      <td>1015.5</td>
      <td>280.0</td>
      <td>7.2</td>
      <td>2016.0</td>
      <td>1.0</td>
      <td>27.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>15.6250</td>
      <td>152.0</td>
      <td>Office</td>
      <td>10301.0</td>
      <td>1970.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>15.3</td>
      <td>NaN</td>
      <td>12.2</td>
      <td>NaN</td>
      <td>1009.1</td>
      <td>210.0</td>
      <td>7.2</td>
      <td>2016.0</td>
      <td>6.0</td>
      <td>29.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>83.3333</td>
      <td>850.0</td>
      <td>Public services</td>
      <td>28590.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>21.7</td>
      <td>2.0</td>
      <td>1.1</td>
      <td>0.0</td>
      <td>1020.2</td>
      <td>350.0</td>
      <td>4.6</td>
      <td>2016.0</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>600.3800</td>
      <td>79.0</td>
      <td>Office</td>
      <td>36240.0</td>
      <td>2010.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>20.6</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>1015.7</td>
      <td>150.0</td>
      <td>3.6</td>
      <td>2016.0</td>
      <td>11.0</td>
      <td>30.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>134.6130</td>
      <td>1090.0</td>
      <td>Office</td>
      <td>305047.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>19.4</td>
      <td>NaN</td>
      <td>16.1</td>
      <td>-1.0</td>
      <td>1008.5</td>
      <td>190.0</td>
      <td>2.6</td>
      <td>2016.0</td>
      <td>5.0</td>
      <td>24.0</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# call datetime_transformer on df_valid
df_valid = datetime_transformer(df_valid, ['timestamp'])

# print the first 5 rows of df_valid
df_valid.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meter</th>
      <th>meter_reading</th>
      <th>building_id</th>
      <th>primary_use</th>
      <th>square_feet</th>
      <th>year_built</th>
      <th>floor_count</th>
      <th>site_id</th>
      <th>air_temperature</th>
      <th>cloud_coverage</th>
      <th>dew_temperature</th>
      <th>precip_depth_1_hr</th>
      <th>sea_level_pressure</th>
      <th>wind_direction</th>
      <th>wind_speed</th>
      <th>timestamp_year</th>
      <th>timestamp_month</th>
      <th>timestamp_day</th>
      <th>timestamp_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>692.220</td>
      <td>376.0</td>
      <td>Office</td>
      <td>585955.0</td>
      <td>1942.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>13.9</td>
      <td>2.0</td>
      <td>7.2</td>
      <td>0.0</td>
      <td>1022.8</td>
      <td>350.0</td>
      <td>6.2</td>
      <td>2016.0</td>
      <td>11.0</td>
      <td>4.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>475.000</td>
      <td>883.0</td>
      <td>Education</td>
      <td>399331.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>36.7</td>
      <td>2.0</td>
      <td>22.2</td>
      <td>0.0</td>
      <td>1014.8</td>
      <td>NaN</td>
      <td>2.6</td>
      <td>2016.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>183.404</td>
      <td>237.0</td>
      <td>Public services</td>
      <td>101262.0</td>
      <td>1982.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>33.3</td>
      <td>0.0</td>
      <td>2.8</td>
      <td>0.0</td>
      <td>1002.4</td>
      <td>240.0</td>
      <td>4.1</td>
      <td>2016.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.000</td>
      <td>176.0</td>
      <td>Education</td>
      <td>62238.0</td>
      <td>1970.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>29.4</td>
      <td>4.0</td>
      <td>-5.6</td>
      <td>0.0</td>
      <td>1012.1</td>
      <td>290.0</td>
      <td>4.1</td>
      <td>2016.0</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>171.597</td>
      <td>755.0</td>
      <td>Office</td>
      <td>42129.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>27.8</td>
      <td>0.0</td>
      <td>21.7</td>
      <td>0.0</td>
      <td>1018.2</td>
      <td>NaN</td>
      <td>2.1</td>
      <td>2016.0</td>
      <td>8.0</td>
      <td>18.0</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
</div>



## Handling missing data

### Identifying missing values


```python
def nan_checker(df):
    """
    The NaN checker

    Parameters
    ----------
    df : dataframe
    
    Returns
    ----------
    The dataframe of variables with NaN, their proportion of NaN and dtype
    """
    
    # get the variables with NaN, their proportion of NaN and dtype
    df_nan = pd.DataFrame([[var, df[var].isna().sum() / df.shape[0], df[var].dtype]
                           for var in df.columns if df[var].isna().sum() > 0],
                          columns=['var', 'proportion', 'dtype'])
    
    # sort df_nan in accending order of the proportion of NaN
    df_nan = df_nan.sort_values(by='proportion', ascending=False)
    
    return df_nan
```


```python
# combine df_train, df_valid and df_test
df = pd.concat([df_train, df_valid], sort=False)
```


```python
# call nan_checker on df
df_nan = nan_checker(df)

# print df_nan
df_nan.reset_index(drop=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var</th>
      <th>proportion</th>
      <th>dtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>floor_count</td>
      <td>1.000000</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>1</th>
      <td>year_built</td>
      <td>0.610350</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cloud_coverage</td>
      <td>0.436067</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>precip_depth_1_hr</td>
      <td>0.188672</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>4</th>
      <td>wind_direction</td>
      <td>0.072537</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sea_level_pressure</td>
      <td>0.061691</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>6</th>
      <td>wind_speed</td>
      <td>0.007231</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>7</th>
      <td>square_feet</td>
      <td>0.005418</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>8</th>
      <td>dew_temperature</td>
      <td>0.005024</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>9</th>
      <td>air_temperature</td>
      <td>0.004849</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>10</th>
      <td>site_id</td>
      <td>0.004554</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>11</th>
      <td>timestamp_year</td>
      <td>0.004554</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>12</th>
      <td>timestamp_month</td>
      <td>0.004554</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>13</th>
      <td>timestamp_day</td>
      <td>0.004554</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>14</th>
      <td>timestamp_hour</td>
      <td>0.004554</td>
      <td>float64</td>
    </tr>
  </tbody>
</table>
</div>



- According to the table, the floor_count has many missing values so we can remove this column. Moreover, the year_built also has high proportion of missing values, and since it is impossible to impute data for this variable we will also remove the column from dataframe.
- We remove missing values in any float64 columns with the proportion of missing data smaller than 5%.
- For the other missing values, we will use different techniques to impute them.

### Dropping columns with high proportion of missing values


```python
# dropping columns in training set
df_train = df_train.drop(columns = ["floor_count","year_built"])
# dropping columns in validation set
df_valid = df_valid.drop(columns = ["floor_count","year_built"])
```


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meter</th>
      <th>meter_reading</th>
      <th>building_id</th>
      <th>primary_use</th>
      <th>square_feet</th>
      <th>site_id</th>
      <th>air_temperature</th>
      <th>cloud_coverage</th>
      <th>dew_temperature</th>
      <th>precip_depth_1_hr</th>
      <th>sea_level_pressure</th>
      <th>wind_direction</th>
      <th>wind_speed</th>
      <th>timestamp_year</th>
      <th>timestamp_month</th>
      <th>timestamp_day</th>
      <th>timestamp_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>64.3250</td>
      <td>1403.0</td>
      <td>Lodging/residential</td>
      <td>78438.0</td>
      <td>15.0</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>-4.4</td>
      <td>NaN</td>
      <td>1015.5</td>
      <td>280.0</td>
      <td>7.2</td>
      <td>2016.0</td>
      <td>1.0</td>
      <td>27.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>15.6250</td>
      <td>152.0</td>
      <td>Office</td>
      <td>10301.0</td>
      <td>1.0</td>
      <td>15.3</td>
      <td>NaN</td>
      <td>12.2</td>
      <td>NaN</td>
      <td>1009.1</td>
      <td>210.0</td>
      <td>7.2</td>
      <td>2016.0</td>
      <td>6.0</td>
      <td>29.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>83.3333</td>
      <td>850.0</td>
      <td>Public services</td>
      <td>28590.0</td>
      <td>8.0</td>
      <td>21.7</td>
      <td>2.0</td>
      <td>1.1</td>
      <td>0.0</td>
      <td>1020.2</td>
      <td>350.0</td>
      <td>4.6</td>
      <td>2016.0</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>600.3800</td>
      <td>79.0</td>
      <td>Office</td>
      <td>36240.0</td>
      <td>0.0</td>
      <td>20.6</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>1015.7</td>
      <td>150.0</td>
      <td>3.6</td>
      <td>2016.0</td>
      <td>11.0</td>
      <td>30.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>134.6130</td>
      <td>1090.0</td>
      <td>Office</td>
      <td>305047.0</td>
      <td>13.0</td>
      <td>19.4</td>
      <td>NaN</td>
      <td>16.1</td>
      <td>-1.0</td>
      <td>1008.5</td>
      <td>190.0</td>
      <td>2.6</td>
      <td>2016.0</td>
      <td>5.0</td>
      <td>24.0</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>



### Removing rows of variables with small proportion of missing values


```python
# get the dataframe of variables that we will remove their missing values
df_miss = df_nan[((df_nan['dtype'] == 'float64') & (df_nan["proportion"] < 0.05))].reset_index(drop=True)
```


```python
df_miss # print df_miss
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var</th>
      <th>proportion</th>
      <th>dtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>wind_speed</td>
      <td>0.007231</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>1</th>
      <td>square_feet</td>
      <td>0.005418</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dew_temperature</td>
      <td>0.005024</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>air_temperature</td>
      <td>0.004849</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>4</th>
      <td>site_id</td>
      <td>0.004554</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>5</th>
      <td>timestamp_year</td>
      <td>0.004554</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>6</th>
      <td>timestamp_month</td>
      <td>0.004554</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>7</th>
      <td>timestamp_day</td>
      <td>0.004554</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>8</th>
      <td>timestamp_hour</td>
      <td>0.004554</td>
      <td>float64</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remove rows with missing values from df_train
df_train = df_train.dropna(subset=np.intersect1d(df_miss['var'], df_train.columns))

# Remove rows with missing values from df_valid
df_valid = df_valid.dropna(subset=np.intersect1d(df_miss['var'], df_valid.columns))


```

### Imputing missing values


```python
df_impute = df_nan[((df_nan['dtype'] == 'float64') & (df_nan["proportion"] > 0.05) & (df_nan["proportion"] < 0.5))].reset_index(drop=True)
```


```python
df_impute # print df_impute
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var</th>
      <th>proportion</th>
      <th>dtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cloud_coverage</td>
      <td>0.436067</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>1</th>
      <td>precip_depth_1_hr</td>
      <td>0.188672</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>wind_direction</td>
      <td>0.072537</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sea_level_pressure</td>
      <td>0.061691</td>
      <td>float64</td>
    </tr>
  </tbody>
</table>
</div>



With the assumption that the current measurement is related to the previous record and the next record, we will impute missing values as the average of the two closest data. In other words, the filling method is the combination of forward fill and backward fill.


```python
for var in df_impute['var']:
    df_train[var] = (df_train[var].fillna(method='ffill', inplace = False) + df_train[var].fillna(method='bfill', inplace = False))/2
    df_valid[var] = (df_valid[var].fillna(method='ffill', inplace = False) + df_valid[var].fillna(method='bfill', inplace = False))/2
```


```python
df_train.isnull().values.any() # re-checking for nan in df_train
```




    True



There are still missing values in df_train. The reason is that the first or last value is missing so the imputing method is not applicable. To solve this issue, we impute missing values for the first rows using backward fill method, and missing values for the last rows using forward fill method. 


```python
df_train[df_train.isnull().any(axis=1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meter</th>
      <th>meter_reading</th>
      <th>building_id</th>
      <th>primary_use</th>
      <th>square_feet</th>
      <th>site_id</th>
      <th>air_temperature</th>
      <th>cloud_coverage</th>
      <th>dew_temperature</th>
      <th>precip_depth_1_hr</th>
      <th>sea_level_pressure</th>
      <th>wind_direction</th>
      <th>wind_speed</th>
      <th>timestamp_year</th>
      <th>timestamp_month</th>
      <th>timestamp_day</th>
      <th>timestamp_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>64.325</td>
      <td>1403.0</td>
      <td>Lodging/residential</td>
      <td>78438.0</td>
      <td>15.0</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>-4.4</td>
      <td>NaN</td>
      <td>1015.5</td>
      <td>280.0</td>
      <td>7.2</td>
      <td>2016.0</td>
      <td>1.0</td>
      <td>27.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>15.625</td>
      <td>152.0</td>
      <td>Office</td>
      <td>10301.0</td>
      <td>1.0</td>
      <td>15.3</td>
      <td>NaN</td>
      <td>12.2</td>
      <td>NaN</td>
      <td>1009.1</td>
      <td>210.0</td>
      <td>7.2</td>
      <td>2016.0</td>
      <td>6.0</td>
      <td>29.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>15895989</th>
      <td>0</td>
      <td>36.000</td>
      <td>895.0</td>
      <td>Education</td>
      <td>55087.0</td>
      <td>9.0</td>
      <td>36.7</td>
      <td>0.0</td>
      <td>22.2</td>
      <td>0.0</td>
      <td>1012.9</td>
      <td>NaN</td>
      <td>2.6</td>
      <td>2016.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
for var in df_impute['var']:
    df_train[var] = df_train[var].fillna(method='ffill', inplace = False)
    df_train[var] = df_train[var].fillna(method='bfill', inplace = False)
```


```python
df_train.isnull().values.any() # re-checking for nan in df_train
```




    False




```python
df_valid.isnull().values.any() # re-checking for nan in df_valid
```




    False



There is no missing value in training set and validation set.

## Changing data type
To reduce memory, we will convert columns with dtypes float64 into float32.


```python
# get a list of columns whose dtypes are float64
df_num_col = df_train.loc[:, df_train.dtypes == 'float64'].columns

# convert datetime components from float64 into float32
for var in df_num_col:
    df_train[var] = df_train[var].astype('float32')
    df_valid[var] = df_valid[var].astype('float32')
```


```python
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 15691522 entries, 0 to 15895989
    Data columns (total 17 columns):
     #   Column              Dtype  
    ---  ------              -----  
     0   meter               object 
     1   meter_reading       float32
     2   building_id         float32
     3   primary_use         object 
     4   square_feet         float32
     5   site_id             float32
     6   air_temperature     float32
     7   cloud_coverage      float32
     8   dew_temperature     float32
     9   precip_depth_1_hr   float32
     10  sea_level_pressure  float32
     11  wind_direction      float32
     12  wind_speed          float32
     13  timestamp_year      float32
     14  timestamp_month     float32
     15  timestamp_day       float32
     16  timestamp_hour      float32
    dtypes: float32(15), object(2)
    memory usage: 1.2+ GB


## Encoding the categorical variables


```python
# combine df_train, df_valid and df_test
df = pd.concat([df_train, df_valid], sort=False)
```

### Identifying categorical variables


```python
def cat_var_checker(df):
    """
    The categorical variable checker

    Parameters
    ----------
    df: the dataframe
    
    Returns
    ----------
    The dataframe of categorical variables and their number of unique value
    """
    
    # Get the dataframe of categorical variables and their number of unique value
    df_cat = pd.DataFrame([[var, df[var].nunique(dropna=False)]
                           for var in df.columns if df[var].dtype == 'object'],
                          columns=['var', 'nunique'])
    
    # Sort df_cat in accending order of the number of unique value
    df_cat = df_cat.sort_values(by='nunique', ascending=False).reset_index(drop=True)
    
    return df_cat
```


```python
# Call cat_var_checker on df
df_cat = cat_var_checker(df)

# Print the dataframe
df_cat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var</th>
      <th>nunique</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>primary_use</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>meter</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



### One-hot encoding the categorical variables


```python
# One-hot-encode the categorical features in the combined data
df = pd.get_dummies(df, columns=np.setdiff1d(df_cat['var'], [target]))

# Print the first 5 rows of df
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meter_reading</th>
      <th>building_id</th>
      <th>square_feet</th>
      <th>site_id</th>
      <th>air_temperature</th>
      <th>cloud_coverage</th>
      <th>dew_temperature</th>
      <th>precip_depth_1_hr</th>
      <th>sea_level_pressure</th>
      <th>wind_direction</th>
      <th>...</th>
      <th>primary_use_Office</th>
      <th>primary_use_Other</th>
      <th>primary_use_Parking</th>
      <th>primary_use_Public services</th>
      <th>primary_use_Religious worship</th>
      <th>primary_use_Retail</th>
      <th>primary_use_Services</th>
      <th>primary_use_Technology/science</th>
      <th>primary_use_Utility</th>
      <th>primary_use_Warehouse/storage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>64.324997</td>
      <td>1403.0</td>
      <td>78438.0</td>
      <td>15.0</td>
      <td>0.600000</td>
      <td>2.0</td>
      <td>-4.4</td>
      <td>0.0</td>
      <td>1015.500000</td>
      <td>280.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.625000</td>
      <td>152.0</td>
      <td>10301.0</td>
      <td>1.0</td>
      <td>15.300000</td>
      <td>2.0</td>
      <td>12.2</td>
      <td>0.0</td>
      <td>1009.099976</td>
      <td>210.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>83.333298</td>
      <td>850.0</td>
      <td>28590.0</td>
      <td>8.0</td>
      <td>21.700001</td>
      <td>2.0</td>
      <td>1.1</td>
      <td>0.0</td>
      <td>1020.200012</td>
      <td>350.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>600.380005</td>
      <td>79.0</td>
      <td>36240.0</td>
      <td>0.0</td>
      <td>20.600000</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>1015.700012</td>
      <td>150.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>134.613007</td>
      <td>1090.0</td>
      <td>305047.0</td>
      <td>13.0</td>
      <td>19.400000</td>
      <td>3.0</td>
      <td>16.1</td>
      <td>-1.0</td>
      <td>1008.500000</td>
      <td>190.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>



### Seperating training and validation sets


```python
# Separating the training data
df_train = df.iloc[:df_train.shape[0], :].copy(deep=True)

# Separating the validation data
df_valid = df.iloc[df_train.shape[0]:df_train.shape[0] + df_valid.shape[0], :].copy(deep=True)
```

## Scaling data

### Transforming target variable using Natural logarithm

According to the [evaluation metric](https://www.kaggle.com/c/ashrae-energy-prediction/overview/evaluation) from the competition, we will apply natural logarithm transformation to the meter_reading variable.


```python
df_train[target] = np.log1p(df_train[target]) # apply log1p transformation to df_train
df_valid[target] = np.log1p(df_valid[target]) # apply log1p transformation to df_valid
```

### Standardizing data

Since the features are measured in various scales, we need to standardize them. We will use StandardScaler to do it.


```python
from sklearn.preprocessing import StandardScaler

# The StandardScaler
ss = StandardScaler()


# Standardize the training data
df_train = pd.DataFrame(ss.fit_transform(df_train), columns=df_train.columns) 

# Standardize the validation data
df_valid = pd.DataFrame(ss.transform(df_valid), columns=df_valid.columns)
```


```python
del df # remove df to save memory
```

## Getting feature matrix and target vector


```python
# Get the feature matrix
X_train = df_train[np.setdiff1d(df_train.columns, [target])]
X_valid = df_valid[np.setdiff1d(df_valid.columns, [target])]


# Get the target vector
y_train = df_train[target]
y_valid = df_valid[target]
```

# Modeling

For this project, we will use 3 models to predict the meter_reading: Linear Regression, Light Gradient Boosting Machine and Multi-layer Perceptron.


```python
# helper functions to upload data to s3
prefix ='sagemaker'

def write_to_s3(filename, bucket, prefix):
    filename_key = filename.split('.')[0]
    key = "{}/{}/{}".format(prefix,filename_key,filename)
    return s3.Bucket(bucket).upload_file(filename,key)

def upload_to_s3(bucket, prefix, filename):
    url = 's3://{}/{}/{}'.format(bucket, prefix, filename)
    print('Writing data to {}'.format(url))
    write_to_s3(filename, bucket, prefix)
```

## Create a dictionary of the models


```python
from sklearn.linear_model import LinearRegression
from lightgbm.sklearn import LGBMRegressor
from sklearn.neural_network import MLPRegressor
```

- In the dictionary:
    - the key is the acronym of the model
    - the value is the model


```python
models = {'lr': LinearRegression(),
          'lgbm': LGBMRegressor(n_estimators=10, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5, random_state=42),
          'mlpr': MLPRegressor(early_stopping=True, random_state=42) }
```

## Create a dictionary of the pipelines

In the dictionary:
- the key is the acronym of the model
- the value is the pipeline, which, for now, only includes the model


```python
from sklearn.pipeline import Pipeline

pipes = {}

for acronym, model in models.items():
    pipes[acronym] = Pipeline([('model', model)])
```

## Hyperparameter tuning and model selection

### Getting the predefined split cross-validator


```python
from sklearn.model_selection import PredefinedSplit

# Combine the feature matrix in the training and validation data
X_train_valid = pd.concat([X_train, X_valid], ignore_index=True, sort=False)

# Combine the target vector in the training and validation data
y_train_valid = pd.concat([y_train, y_valid], ignore_index=True, sort=False)

# Get the indices of training and validation data
train_valid_idxs = np.append(np.full(X_train.shape[0], -1), np.full(X_valid.shape[0], 0))

# The PredefinedSplit
ps = PredefinedSplit(train_valid_idxs)
```

### Creating the dictionary of the parameter grids

- In the dictionary:
    - the key is the acronym of the model
    - the value is the parameter grid of the model


```python
param_grids = {}
```

### The parameter grid for LinearRegression
Since the sklearn LinearRegression is based on the normal equation (a closed-form solution), it is not essential to fine-tune the hyperparameters of the model.


```python
param_grids['lr'] = [{}]
```

### The parameter grid for LightGBM
The hyperparameters we want to fine-tune are:
- learning_rate 
- min_data_in_leaf
- reg_alpha
- reg_lambda


```python
# The grids for learning_rate
learning_rate_grids = [10 ** i for i in range(-3, 0)]

# The grids for min_data_in_leaf, we set high values for big dataset to reduce overfitting
min_data_in_leaf_grids = [100, 1000]

# The grids for reg_alpha
reg_alpha_grids = [10 ** i for i in range(-1, 2)]

# The grids for reg_lambda
reg_lambda_grids = [10 ** i for i in range(-1, 2)]

# Update param_grids
param_grids['lgbm'] = [{'model__learning_rate': learning_rate_grids,
                        'model__min_data_in_leaf': min_data_in_leaf_grids,
                        'model__reg_alpha': reg_alpha_grids,
                        'model__reg_lambda': reg_lambda_grids }]
```

### The parameter grid for MLPRegressor

The hyperparameters we want to fine-tune are:
- learning_rate_init 



```python
# The grids for learning_rate_init
learning_rate_init_grids = [10 ** i for i in range(-3, 0)]

# Update param_grids
param_grids['mlpr'] = [{'model__learning_rate_init': learning_rate_init_grids}]
```

### Creating the directory for GridSearchCV results


```python
from sklearn.model_selection import GridSearchCV

# The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV
best_score_param_estimator_gs = []

for acronym in pipes.keys():
    # GridSearchCV
    gs = GridSearchCV(estimator=pipes[acronym],
                      param_grid=param_grids[acronym],
                      scoring='neg_mean_squared_error',
                      n_jobs=3,
                      cv=ps,
                      return_train_score=True)
        
    # Fit the pipeline
    gs = gs.fit(X_train_valid, y_train_valid)
    
    # Update best_score_param_estimator_gs
    best_score_param_estimator_gs.append([gs.best_score_, gs.best_params_, gs.best_estimator_])
    
    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])
    
    # Get the important columns in cv_results
    important_columns = ['rank_test_score',
                         'mean_test_score', 
                         'std_test_score', 
                         'mean_train_score', 
                         'std_train_score',
                         'mean_fit_time', 
                         'std_fit_time',                        
                         'mean_score_time', 
                         'std_score_time']
    
    # Move the important columns ahead
    cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]
    
    # Write cv_results file
    cv_results.to_csv(path_or_buf = acronym + '.csv', index=False)
    
    # upload results to s3 bucket
    upload_to_s3(bucket, prefix, filename = acronym + '.csv')
```

    Writing data to s3://dataset.finalproject.cc2020/sagemaker/lr.csv
    Writing data to s3://dataset.finalproject.cc2020/sagemaker/lgbm.csv
    Writing data to s3://dataset.finalproject.cc2020/sagemaker/mlpr.csv


## GridSearchCV Results


```python
# Sort best_score_param_estimator_gs in descending order of the best_score_
best_score_param_estimator_gs = sorted(best_score_param_estimator_gs, key=lambda x : x[0], reverse=True)

# Print best_score_param_estimator_gs
pd.DataFrame(best_score_param_estimator_gs, columns=['best_score', 'best_param', 'best_estimator'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>best_score</th>
      <th>best_param</th>
      <th>best_estimator</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.385137</td>
      <td>{'model__learning_rate_init': 0.001}</td>
      <td>(MLPRegressor(activation='relu', alpha=0.0001,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.659932</td>
      <td>{'model__learning_rate': 0.1, 'model__min_data...</td>
      <td>(LGBMRegressor(bagging_fraction=0.8, bagging_f...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.800530</td>
      <td>{}</td>
      <td>(LinearRegression(copy_X=True, fit_intercept=T...</td>
    </tr>
  </tbody>
</table>
</div>



According to the results, MLPRegressor is the best model. LightGBM is the second best one, and Linear Regression has the worst performance.


```python
# Get the best_score, best_param and best_estimator of LightGBM
best_score_lgbm, best_param_lgbm, best_estimator_lgbm = best_score_param_estimator_gs[0]

# Get the best_score, best_param and best_estimator of MLP
best_score_mlp, best_param_mlp, best_estimator_mlp = best_score_param_estimator_gs[1]

# Get the best_score, best_param and best_estimator of Linear Regression
best_score_lr, best_param_lr, best_estimator_lr = best_score_param_estimator_gs[2]
```

## Feature Importance
We will have a look at the features that greatly contribute to the prediction in the best model: Multi-layer Perceptron.


```python
features = np.setdiff1d(df_train.columns, [target]) # getting the names of features

# Get the dataframe of feature and importance
df_fi = pd.DataFrame(np.hstack((features.reshape(-1, 1), best_estimator_mlp.named_steps['model'].feature_importances_.reshape(-1, 1))),
                         columns=['Features', 'Importance'])

# Sort df_fi_rfc in descending order of the importance
df_fi = df_fi.sort_values(ascending=False, by='Importance').reset_index(drop=True)

# Print the first 5 rows of df_fi_rfc
df_fi.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Features</th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>building_id</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>square_feet</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>air_temperature</td>
      <td>28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>meter_3.0</td>
      <td>22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>site_id</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a figure to plot top 15 importance features
fig = plt.figure(figsize=(10, 10))

# The bar plot of the top 15 feature importance
df_fi_top = df_fi.iloc[0:15]
plt.bar(df_fi_top['Features'], df_fi_top['Importance'], color='blue')

# Set x-axis
plt.xlabel('Features')
plt.xticks(rotation=90)

# Set y-axis
plt.ylabel('Importance')

# title
plt.title("Multi-layer Perceptron - Feature Importance", fontsize=15)

# Save and show the figure
plt.tight_layout()
plt.savefig('Feature_Importance_MLP.pdf')
plt.show()


```


![png](output_122_0.png)


The most important predictors in MLP model are building_id, the area of building's floor, the temperature, the energy type and the primary category of activities for the building. Moreover, time series component also contributes to the forecast of energy consumption.


```python
# upload figure to s3 bucket
upload_to_s3(bucket, prefix, filename = 'Feature_Importance_MLP.pdf')
```

    Writing data to s3://dataset.finalproject.cc2020/sagemaker/Feature_Importance_MLP.pdf


# Conclusion

<div style="text-align: justify">This project proposes the utilization of AWS services in a real big data problem. S3 bucket gives flexibility to store and load data without worrying about memory usage. AWS Athena with friendly SQL syntax and queries is useful for overall data exploration. The JOIN command makes it easier to work with multiple datasets and features. AWS Sagemaker provides many benefits for preprocessing big data and building complex models . With the help of AWS Sagemaker, it is easier to handle more than 20 million observations and 15 features from 3 datasets. Moreover, the computation power from Sagemaker instance helps to accelerate the training and hyper-parameter tuning tasks which would last hours in a local machine.</div>


```python

```
