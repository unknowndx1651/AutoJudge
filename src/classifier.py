from src import feature_extraction,preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import hstack
import pandas as pd;
import joblib as jb;

def get_target(labels):
    mappings={"easy":0, "medium":1, "hard":2}
    mapped_label=[];
    for label in labels:
        mapped_label.append(mappings[label]);
    return np.array(mapped_label);

def data_split(texts, target):
    xtrain,xtest,ytrain,ytest=train_test_split(texts,target, train_size=0.80,stratify=target);
    return (xtrain,xtest,ytrain,ytest);

tfidf=feature_extraction.get_vectoriser();
def training_data(X_train):
    
    X_train_1=tfidf.fit_transform(X_train);
    X_train_2=feature_extraction.features(X_train);
    
    return hstack([X_train_1, X_train_2]);
    
def testing_data(X_test):
    
    X_test1=tfidf.transform(X_test);
    X_test2=feature_extraction.features(X_test);
    
    return hstack([X_test1, X_test2]);

#feature generation:

dataset=pd.read_csv("data/problems_data.csv");
Y=get_target(dataset["problem_class"])
preprocess=preprocessing.preprocessing(dataset)
X=preprocess["combined_text"]

xtrain,xtest,ytrain,ytest=data_split(X,Y);

xtrain_final=training_data(xtrain);
xtest_final=testing_data(xtest);


#Classification Model:

from sklearn.ensemble import RandomForestClassifier;
classifier=RandomForestClassifier(n_estimators=700, class_weight="balanced")

classifier.fit(xtrain_final,ytrain)
ypred=classifier.predict(xtest_final);


jb.dump(tfidf, "models/vectoriser.pkl");
jb.dump(classifier, "models/RFC.pkl")