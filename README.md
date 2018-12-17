# CS410 Final Project  - Personalized App Search
By YoungJu Kwon

## Documenation:
Summary
This application is to search personalized app based on the app installation information. App embedding is created through using word2vec algorithm, which is believed to show better presentations of apps’ characteristics than their categories. Combined with the embedding, the project will retrieve personalized app search results for the query given. Other than that, it would suggest basic search result for cold start.


### Data
Two dataset is used for 
#### App Data
Original data is collected through 8.01 through 11.01 on moblie application, it consists of individual random id, what application the person installed, and the date the person opened. \
ADID: Random individual id \
Date: Date the individual opened the application \
App: app key number denoted \
![](vector tsne.png?raw=true)

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

### Functions


File description
functions.py
create_vector_model.ipynb
spark_process_data.Rmd
stopwords.txt
stopwords_k.txt
