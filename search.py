import sys
import pandas as pd
import numpy as np
from functions import *

# inverted index
import collections
import json

# tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

# word2vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

class app_search:
    
    def __init__(self):
        # read meta
        self.meta = pd.read_csv("data/android_meta.csv")

        # read inverted index
        self.inv_idx = json.load(open('data/inv_idx'))
        
        # read corpus
        self.corpus = self.meta.iloc[:,7]
            
        # each document length
        self.doc_len = pd.DataFrame([len(text.split()) for text in self.corpus])
        
        # vectorizer
        self.vectorizer = TfidfVectorizer(min_df=1)
        self.tf_idf_matrix = self.vectorizer.fit_transform(self.meta.appTitle)
        
        # app embeddings
        self.model = KeyedVectors.load_word2vec_format('data/item_vectors_12iter.txt', binary=False)
        
        # set vector_list None
        self.user_app_list = 0
        self.mean_vec = 0
     
    def app_list(self, input_list):
        # check whether word exists in model
        input_list = [str(app) for app in input_list]
        input_list = list(filter(lambda x: x in self.model.vocab, input_list ))
        
        # save list
        self.user_app_list = input_list
        # save mean vector of user
        self.mean_vec = np.mean([self.model.word_vec(app) for app in  input_list],0)
        
            
    def vector_similarity(self, search_result):
        
        mean_vec = self.mean_vec
        model = self.model
        search_result['vec_sim']=0
        for app_key in search_result.app_key:
            try:
                app_vec = model.word_vec(str(app_key))
                vec_sim = np.matmul(app_vec, mean_vec)/(np.linalg.norm(app_vec)*np.linalg.norm(mean_vec))
                idx = search_result['app_key']==app_key
                search_result.ix[idx,'vec_sim']=vec_sim
                

            except:
                pass
            
        return search_result


    def search(self, query, n = 20):
        
        # get all needed
        vectorizer = self.vectorizer
        tf_idf_matrix = self.tf_idf_matrix
        meta = self.meta
        
        # create tf-idf matrixs
        tf_result = tfidf_search(query, vectorizer, tf_idf_matrix)

        # get bm results
        bm_result = bm25(query,self.inv_idx, self.doc_len)
        
        # merge bm and tf-idf results
        result = bm_result.merge(tf_result[tf_result.score>0.5], how="outer", on="idx")
        result['app_key']=meta.ix[result.idx].app_key.values
        # merge meta data
        result = result.merge(meta, on="app_key", how="left")

        # conver Nan to zero
        bm_ratio = 0.02
        result['score'] = result['score'].fillna(0)
        result['bm_score'] = result['bm_score'].fillna(0)
        # get mixed score of bm and tfidf
        result['mix']= result['bm_score']*bm_ratio+result['score']
        sorted_result = result.sort_values('mix', ascending=False)
        
        if(self.user_app_list !=0):
            vec_result = self.vector_similarity(sorted_result)
            vec_result['mix']= vec_result['bm_score']*bm_ratio+vec_result['score']+vec_result['vec_sim']
            return vec_result[:n]
        else:
            return sorted_result.loc[:,['appTitle', 'app_id']][:n]


    