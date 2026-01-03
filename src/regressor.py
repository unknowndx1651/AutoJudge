from src import feature_extraction,preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import hstack
import pandas as pd;
import joblib as jb;

tfidf=jb.load("models/vectoriser.pkl")  

def get_target(scores):
    return np.array(scores)

def training_data(X_train):
    
    X_train1=tfidf.transform(X_train);
    X_train2=feature_extraction.features(X_train);
    
    return hstack([X_train1, X_train2]);

def testing_data(X_test):
    
    X_test1=tfidf.transform(X_test);
    X_test2=feature_extraction.features(X_test);
    
    return hstack([X_test1, X_test2]);

def data_split(texts, target):
    xtrain,xtest,ytrain,ytest=train_test_split(texts,target, train_size=0.80);
    return (xtrain,xtest,ytrain,ytest);

#feature generation 

dataset=pd.read_csv("data/problems_data.csv");
Y=get_target(dataset["problem_score"])
preprocess=preprocessing.preprocessing(dataset)
X=preprocess["combined_text"]

xtrain,xtest,ytrain,ytest=data_split(X,Y);

xtrain_final=training_data(xtrain);
xtest_final=testing_data(xtest);



#Regression Model:

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100)

regressor.fit(xtrain_final,ytrain)
ypred=regressor.predict(xtest_final);

jb.dump(regressor, "models/RFR.pkl")