# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 12:14:53 2018

@author: SenKa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def readData(fpath):
    df = pd.read_csv(fpath)
    return df

def main():
    df1 = readData("../data/complete_dataset.csv")
    df2 = readData("../data/complete_dataset2.csv")
    df3 = readData("../data/complete_dataset3.csv")
    df = df3.copy() #pd.concat([df1,df2])
    testNames = np.unique(df.testName)
    dfUS = df[df.participantCountryOfResidence == "US"]
    dfGB = df[df.participantCountryOfResidence == "GB"]
    dfAU = df[df.participantCountryOfResidence == "AU"]
    #grp = df.groupby()
    

    
    dfUSControl = dfUS[dfUS.participantIsControl == True].copy()
    dfUSTreated = dfUS[dfUS.participantIsControl == False].copy()
    dfUSControl.drop(columns = ['testResultMetricId', 'testResultMetricCreatedOn'], inplace = True)
    dfUSTreated.drop(columns = ['testResultMetricId', 'testResultMetricCreatedOn'], inplace =True)
    #dfUSControl.drop_duplicates(inplace = True)
    #dfUSTreated.drop_duplicates(inplace = True)
    dfUSControlTest = pd.DataFrame(pd.pivot_table(dfUSControl, index = "floodlightOpenId", 
                                                  columns = "testMetricName", 
                                                  values = "testResultMetricValue",
                                                  aggfunc = 'max').to_records())
    dfUSTreatedTest = pd.DataFrame(pd.pivot_table(dfUSTreated, index = "floodlightOpenId", 
                                                  columns = "testMetricName", 
                                                  values = "testResultMetricValue",
                                                  aggfunc = 'max').to_records())
    
    dfUSControlTest = pd.DataFrame(pd.pivot_table(dfUSControl, index = "floodlightOpenId", 
                                                  columns = "testMetricName", 
                                                  values = "testResultMetricValue",
                                                  ).to_records())
    dfUSTreatedTest = pd.DataFrame(pd.pivot_table(dfUSTreated, index = "floodlightOpenId", 
                                                  columns = "testMetricName", 
                                                  values = "testResultMetricValue",
                                                  ).to_records())
    dfUSControlTest.fillna(0, inplace = True)
    dfUSTreatedTest.fillna(0, inplace = True)
    dfUSControlTest.insert(0,"id", range(0, len(dfUSControlTest)))
    dfUSTreatedTest.insert(0,"id", range(0, len(dfUSTreatedTest)))
    #dfUSTreatedTest["class"] = 1
    #dfUSControlTest["class"] = 2
       
    clsMS = dfUSTreatedTest.copy()
    clsNoMS = dfUSControlTest.copy()    
    clsMS.drop(columns = ["id", "floodlightOpenId"], inplace = True)
    clsNoMS.drop(columns = ["id", "floodlightOpenId"], inplace = True)
    
     #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    
    ## Normalizing the data
    x = clsMS.values
    x_scaled = min_max_scaler.fit_transform(x)
    clsMS_scaled = pd.DataFrame(x_scaled)
    
    x = clsNoMS.values
    x_scaled = min_max_scaler.fit_transform(x)
    clsNoMS_scaled = pd.DataFrame(x_scaled)
    
    ## Preparation of train and test data ..   
    X1 = clsMS_scaled.as_matrix()
    X2 = clsNoMS_scaled.as_matrix() 
    X = np.concatenate((X1,X2))
    y1 = np.zeros(len(clsMS))
    y2 = np.ones(len(clsNoMS))
    
    ## No scaling 
    
    X1 = clsMS.as_matrix()
    X2 = clsNoMS.as_matrix() 
    X = np.concatenate((X1,X2))
    y1 = np.zeros(len(clsMS))
    y2 = np.ones(len(clsNoMS))
    
    
    pca = True
    components = 2
       
    
    if not pca:
    ## Without PCA
        X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.1, random_state=42)
        X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.1, random_state=42) 
        X_train = np.concatenate((X1_train,X2_train))
        y_train = np.concatenate((y1_train,y2_train))
        
        X_test = np.concatenate((X1_test,X2_test))
        y_test = np.concatenate((y1_test,y2_test)) 
        
        ## Without PCA
        F_train = X_train
        l_train = y_train
        
        F_test = X_test
        l_test = y_test
        
    if pca:        
        ## With PCA
        X1_r = PCA(n_components = components).fit(X1).transform(X1)
        X2_r = PCA(n_components = components).fit(X2).transform(X2)
        X1_train, X1_test, y1_train, y1_test = train_test_split(X1_r, y1, test_size=0.1, random_state=42)
        X2_train, X2_test, y2_train, y2_test = train_test_split(X2_r, y2, test_size=0.1, random_state=42)
        
        X_r = np.concatenate((X1_train,X2_train))
        y1_r = np.zeros(len(X1_train))
        y2_r = np.ones(len(X2_train))
        y_r = np.concatenate((y1_train,y2_train))
        X1_r = np.delete(X1_train, 42, 0)  ## outlier
        
        ## With PCA
        F_train = X_r
        l_train = y_r
        
   
    fitmodel(X,y)

def fitmodel(F, l):
    qda = QuadraticDiscriminantAnalysis()
    qmod= qda.fit(F_train,l_train)    
    qmod.predict(X1_test)
    qmod.predict(X2_test)
    
    dtree = tree.DecisionTreeClassifier().fit(F_train,l_train)
    dtree.predict(X1_test)
    dtree.predict(X2_test)
    
    rdc = RandomForestClassifier().fit(F_train,l_train)
    rdc.predict(X1_test)
    rdc.predict(X2_test)
        
    
def plotFeatures(df1, df2, xaxis = "", featurenames = [""]):
    
    for feature in featurenames:
        #plt.figure()
        ax = df1.plot(x = xaxis, y = feature, color = 'b', kind = 'scatter', title = feature)
        df2.plot(x = xaxis, y = feature, color = 'r', kind = 'scatter', ax = ax)
    

if __name__ == '__main__':
    main()