# McMaster University Spectral Irradiance Centre (MUSIC)
![](/images/roof.png)

**This is my pet project which consists of analysis of spectral irradiance and weather data provided by MUSIC 
and performing various forcasting models.**

# Goal
The goal of this project is to perform a comprehensive analysis of data provided by MUSIC and learn forecasting techniques. 
The data consists of raw and precomputed measurements of meteorological and spectral irradiance data. The first part of the project is to 
explore the data, perform data cleaning and compare the experimental dataset with relevant external datasets to verify the correctness of
experimental measurements. The second part is to preform univariate and multivariate forecasting using various statistical and
machine learning techniques (ARIMA, Linear regression, Trees, LSTM) and compare the results. 

# Data
The data consists of time series with about 30 columns of measurements acquired from MUSIC every minute. All the data is divided into 2 different parts:
meteorological provided by weather station WS600, and all the spectral data provided by varios spectral detectors. 
The time frame of the acquired data is: July, 1st 2019 - December, 17th 2019.
Small part of the raw data is available [here](https://github.com/Ruslanmalai/MUSIC/tree/master/data/Daily%20Summary%20Files).

### Weather Station (WS600)
Weather station WS600 provides data about ambient temperature, pressure, humidity, precipitation, wind speed and wind direction.

Some of the weather measurements are also taken by spectral detectors. Readings from the decectors are shown in the EDA. However, measurements
from the weather station seem to be more accurate and trustworthy. Thus, in future forecasting the detectors weather measurements are not taken
into consideration.

**External Dataset**. The external dataset is taken from this [website](https://rp5.ru/Weather_archive_in_Hamilton_(airport),_Canada,_METAR). It is also available in the [repo](https://github.com/Ruslanmalai/MUSIC/blob/master/data/external%20data.xls).


**NASA Ozone data**. The external dataset with daily ozone level is taken from the [NASA Ozone Watch](https://ozonewatch.gsfc.nasa.gov/monthly/SH.html) database. The NASA data is compared with the experimental ozone data in the
Data Exploration section.

### Spectral Irradiance
The spectral irradiance data is provided by a set of different spectral detectors:
*To be continued*

# Data exploration and analysis

**Please, use this [link](https://nbviewer.jupyter.org/github/Ruslanmalai/MUSIC/blob/master/EDA.ipynb) to open Data Analysis notebook**

*In progress*

# Forecasting

## Univariate Forecasting

*In progress*

## Multivariate Forecasting

*In progress*
