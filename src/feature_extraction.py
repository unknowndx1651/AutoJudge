import numpy as np;
from sklearn.feature_extraction.text import TfidfVectorizer;
from scipy.sparse import hstack, csr_matrix;
def get_vectoriser():
    tfidf=TfidfVectorizer( ngram_range= (1,2),
                          stop_words="english",
                          max_features=5000
                          );
    return tfidf;

def fit_transform_text(tfidf:TfidfVectorizer, texts):
    X=tfidf.fit_transform(texts);
    return X;
    
#other features:

keyword_list=["dp", "graph","tree", "array", "subarray", "recursion", "greedy", "dfs", "bfs",
              "subsequence", "matrix", "multiset", "prefix", "suffix"]
math_symbols=['+', '-', '*', '/', '=', '<', '>', '^' ,'%']

def count_symbols(text:str):
    sym_freq=0
    for sym in math_symbols:
        sym_freq = sym_freq + text.count(sym);
    return sym_freq
 
def length(text:str):
    return len(text.split());

def count_kw(text:str):
    kw_freq=0
    for kw in keyword_list:
        kw_freq = kw_freq + text.count(kw);
    return kw_freq
 
def features(texts):
    feature=[];
    for text in texts:
        length_feature=len(text);  
        keyword_feature=count_kw(text);
        math_feature=count_symbols(text);
    
        feature.append([length_feature, keyword_feature, math_feature])
    return csr_matrix(np.array(feature));

def X_final(X_tfidf, texts):
    X=hstack([X_tfidf, features(texts)]);
    return X;


        
