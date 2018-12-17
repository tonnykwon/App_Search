# CS410 Final Project  - Personalized App Search
By YoungJu Kwon

## Documenation:
Summary
This application is to search personalized app based on the app installation information. App embedding is created through using word2vec algorithm, which is believed to show better presentations of apps’ characteristics than their categories. Combined with the embedding, the project will retrieve personalized app search results for the query given. Other than that, it would suggest basic search result for cold start.


### Data
https://www.dropbox.com/s/fvkcn0szmmb4y9j/data.zip?dl=0 \
Two dataset is used for for searching app.
#### App Data
Original data is collected through 8.01 through 11.01 on moblie application, it consists of individual random id, what application the person installed, and the date the person opened. It has about 6764,000 instances. \
ADID: Random individual id \
Date: Date the individual opened the application \
App: app key number denoted \

#### Meta Data
Meta Data is about each app. It contains app description, app title, app categories, and each app's key number, which is same as app data's key number. It is collected on 10.31. \
app_key: app key \
app_id: ID of application on play store \
categorycode1: category 1 \
categorycode2: category 2\
categoryname: name of category\
appTitle: name of app\
description: description on app page\
developer: developer name of application\

### Files and functions
#### spark_process_data.Rmd
This R code file is to process raw App data to train and test data for training word2vec of applications. Based on date, the app installed later becomes target, seperated from currently installed app list. sparklyr package is used for handling relative big data. \

#### create_vector_model.ipynb
This notebook is to train and test word2vec through data processed by spark_process_data. After training word2vec algorithm with embedding size=200, negative sampling =15, with negative exponent=0.75, learning_rate = 0.04, and iteration=12. These parameters are tuned through predicting next app installation with k-nearest neighbors. The precision is measure on mean average precision on when k=10, 30, 50. It showed approximately 3%~4%. Considering there are about 50,000 different applications, it seems to be trained well.

#### functions.py
This python file contains functions needed to preprocess app description file. 

File description
functions.py
spark_process_data.Rmd
stopwords.txt
stopwords_k.txt


## Install and run
