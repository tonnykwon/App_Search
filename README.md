# CS410 Final Project  - Personalized App Search
By YoungJu Kwon

## Documenation:
### Summary
This application is to search personalized app based on the app installation information. App embedding is created through using word2vec algorithm, which is believed to show better presentations of apps’ characteristics than their categories. Combined with the embedding, the project will retrieve personalized app search results for the query given. Other than that, it would suggest basic search result for cold start.


### Data
Download: \
https://www.dropbox.com/s/fvkcn0szmmb4y9j/data.zip?dl=0 \
Two dataset is used for for searching app.
#### App Data
Original data is collected through 8.01 through 11.01 on moblie application, it consists of individual random id, what application the person installed, and the date the person opened. It has about 6764,000 instances. \
ADID: Random individual id \
Date: Date the individual opened the application \
App: app key number denoted 

#### Meta Data
Meta Data is about each app. It contains app description, app title, app categories, and each app's key number, which is same as app data's key number. It is collected on 10.31. \
app_key: app key \
app_id: ID of application on play store \
categorycode1: category 1 \
categorycode2: category 2\
categoryname: name of category\
appTitle: name of app\
description: description on app page\
developer: developer name of application

### Files and functions
#### spark_process_data.Rmd
This R code file is to process raw App data to train and test data for training word2vec of applications. Based on date, the app installed later becomes target, seperated from currently installed app list. sparklyr package is used for handling relative big data.

#### create_vector_model.ipynb
This notebook is to train and test word2vec through data processed by spark_process_data. After training word2vec algorithm with embedding size=200, negative sampling =15, with negative exponent=0.75, learning_rate = 0.04, and iteration=12. These parameters are tuned through predicting next app installation with k-nearest neighbors. The precision is measure on mean average precision on when k=10, 30, 50. It showed approximately 3%~4%. Considering there are about 50,000 different applications, it seems to be trained well.

#### functions.py
This python file contains functions needed to preprocess app description file. \
`stopword(corpus, path)`: this function takes stopword path, and remove stopwords such as '%', '.', or other unnecessary letters. \
`inverted_index(corpus, min=1, max=10)`: this function creates inverted index of corpus by using collecionts defaultdict packages. min and max 
specify min or max length of words. It returns dictionary of word and index list. For instance, {'word':[1,3, 5,12,...]}. \
`bm25(query, inv_idx, doc_len, k=2.7, b=0.75)`: bm25 gets bm25 score of each document(*app description*) for the given query. inv_idx is inverted index created from inverted_index function, while k, and b are parameters. It returns pandas dataframe of two columns, index of result, and score. \
`tfidf_search(query, vectorizer, tf_idf_matrix)`:tfidf_search gets score of each document(*app title*) for the given query.

#### search.py
This file contains app_search class, which is used for actual searching. \
`search(self, query, n = 20)`: search function uses bm25, and tfidf_search for ranking each app description and app title, and get overall scores of each document. It returns search result of sorted score. \
`app_list(self, input_list)`: app_list takes list of app key number(which is on meta data), and gets the mean vector value of app lists. This mean value vector represents user's preference, which is sued for search function if app list exists.

## Install and run
Install python 3 on the website: \
https://www.python.org/downloads/ \

Upgrade pip on command line.
``` cmd
pip install --upgrade pip
```
Then, install `sklearn`, `gensim` packages.
``` cmd
pip install sklearn
pip install gensim
```
And download github files.
```cmd
git clone https://github.com/tonnykwon/App_Search
```

Download dataset from dropbox. \
https://www.dropbox.com/s/fvkcn0szmmb4y9j/data.zip?dl=0 \
Unzip data file, and put data folder in data folder into App_search folder. \
For instance, data should look like `App_search/data/android_meta.csv`.

Go back to command line, and start search python script. \
Then load app_search class, this will take some time as it load all inverted index, and meta data.
``` python
python -i search.py
s = app_search()
```

Now, we are ready to put query and search. Use search function to find applications.
``` python
s.search('facebook')
```
![](facebook.png)









