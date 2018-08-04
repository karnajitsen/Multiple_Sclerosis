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


def readData(fpath):
    df = pd.read_csv(fpath)
    return df

def main():
    df1 = readData("../data/complete_dataset.csv")
    df2 = readData("../data/complete_dataset2.csv")
    df = pd.concat([df1,df2])
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
    
    X1 = clsMS.as_matrix()
    X2 = clsNoMS.as_matrix()
    X = np.concatenate((X1,X2))
    y1 = np.ones(len(clsMS))
    y2 = np.ones(len(clsNoMS)) *2
    y = np.concatenate((y1,y2))
    pca1 = PCA(n_components = 2).fit(X1)
    pca2 = PCA(n_components = 2).fit(X2)
    
    X1_r = PCA(n_components = 18).fit(X1).transform(X1)
    X2_r = PCA(n_components = 18).fit(X2).transform(X2)
    
    X_r = np.concatenate((X1_r,X2_r))
    y1_r = np.ones(len(X1_r))
    y2_r = np.ones(len(X2_r)) *2
    y_r = np.concatenate((y1_r,y2_r))
    X1_r = np.delete(X1_r, 42, 0)  ## outlier

    F = X_r
    l = y_r   
    
    plt.scatter(X1_r[:, 0], X1_r[:, 1], color=colors[0], alpha=.8, lw=lw,label='MS')
    
    plt.scatter(X2_r[:, 0], X2_r[:, 1], color=colors[1], alpha=.8, lw=lw,label='No MS')

    
    
    #plt.legend(loc='best', shadow=False, scatterpoints=1)
    #plt.title('Multiple Sclerosis Dataset')
    
    
    
    #plt.show()
    
    
    
    fitmodel(X,y)

def fitmodel(F, l):
    qda = QuadraticDiscriminantAnalysis()
    qmod= qda.fit(F,l)    
    qmod.predict(F)
    
    dtree = tree.DecisionTreeClassifier().fit(F,l)
    dtree.predict(F)
    
    rdc = RandomForestClassifier().fit(F,l)
    rdc.predict(F)
    
    
    

    
    
def plotFeatures(df1, df2, xaxis = "", featurenames = [""]):
    
    for feature in featurenames:
        #plt.figure()
        ax = df1.plot(x = xaxis, y = feature, color = 'b', kind = 'scatter', title = feature)
        df2.plot(x = xaxis, y = feature, color = 'r', kind = 'scatter', ax = ax)
    

if __name__ == '__main__':
    main()