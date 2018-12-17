import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import collections

def stopword(corpus, path):
    """
    remove stopwords
    
    params
    corpus: text data to remove stop words
    path: stopword.txt path
    
    return
    result: filtered corpus(list)
    """
    
    # split line into individual words and remove words
    with open(path) as txtfile:
        stop_list= [line.rstrip() for line in txtfile]
    corpus = [' '.join([word for word in line.split() if word not in stop_list]) for line in corpus]
    
    # remove individaul stop word
    stop_dict={}
    with open('stopwords.txt') as txtfile:
        for line in txtfile:
            key = line.split()
            stop_dict[key[0]] = ''
            
    result = [multireplace(line, stop_dict) for line in corpus]
    
    return result

def multireplace(string, replacements):
    """
    Given a string and a replacement map, it returns the replaced string.
    param
    string: string to execute replacements on
    replacements: replacement dictionary {value to find: value to replace}

    return: replaced string
    """
    substrs = sorted(replacements, key=len, reverse=True)

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)))

    # For each match, look up the new string in the replacements
    return regexp.sub(lambda match: replacements[match.group(0)], string)


def inverted_index(corpus, min=1, max=10):
    """
    Create inverted index
    
    params
    corpus: text data for inverted_idx
    min: minium length of word
    max: maximum length of word
    
    return
    inv_idx: inverted index of corpus
    """
    from collections import defaultdict
    
    inv_idx = defaultdict(list)
    
    # Gather index and text
    [[inv_idx[word].append(idx) for word in text.split() if (len(word)>min and len(word)<max) ] for idx, text in enumerate(corpus)]
        
    return inv_idx
        

def bm25(query, inv_idx, doc_len, k=2.7, b=0.75):
    """
    get bm25 score for app description
    
    params
    query: query input
    inv_idx: inverted index of corpus
    k,b: parameters for bm25
    
    return
    bm_result: score result of bm25
    """
    
    result=pd.DataFrame()
    words = query.split()
    for word in words:
        # in case does not have query word
        try:
            # get frequency of words
            counter=collections.Counter(inv_idx[word])
            # get length of document containing query
            query_doc_len = doc_len.ix[counter.keys()]
            # get frequency of query in document
            word_count = np.array(list(counter.values()))
            
            # bm25 calculation
            normalized_doc = 1-b+b*query_doc_len[0]/np.mean(doc_len[0])
            idf = np.log((len(doc_len) - len(query_doc_len)+0.5 )/(len(query_doc_len)+0.5 ))
            denom = word_count+(k*normalized_doc)
            nom = (k+1)*word_count
            result = result.append(nom/denom*idf, ignore_index=True)
            
            # to pandas with idx, values
            result = np.sum(result)
            bm_result = pd.DataFrame({"idx":result.index, "bm_score":result})
        except:
            pass
        
    return bm_result


def tfidf_search(query, vectorizer, tf_idf_matrix):
    """
    get tf-idf score for app title
    
    params
    query: query input
    vectorizer: TfidfVectorizer fitted for app titles
    tf_idf_matrix: matrix fitted by vectorizer
    
    return
    result: score result of tf-idf
    """
    
    # transform query to tf-idf matrix
    query_mat = vectorizer.transform(query.split())
    # get score and id by multiplying and sorting
    idx = np.argsort(-np.sum(query_mat*tf_idf_matrix.T, 0))
    idx = np.squeeze(np.asarray(idx))
    score = np.sort(-np.sum(query_mat*tf_idf_matrix.T, 0))
    score = np.squeeze(np.asarray(score))
    # to dataframe
    result = pd.DataFrame({"idx":idx, 'score':-score})
    
    return result



