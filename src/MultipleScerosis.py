import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def readData(fpath):
    df = pd.read_csv(fpath)
    return df

def main():
    df = readData("../data/complete_dataset.csv")
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
    dfUSTreatedTest["class"] = 1
    dfUSControlTest["class"] = 2
    fitmodel(dfUSTreatedTest, dfUSControlTest)

def fitmodel(c1, c2):
    clf = LinearDiscriminantAnalysis()
    
def plotFeatures(df1, df2, xaxis = "", featurenames = [""]):
    
    for feature in featurenames:
        #plt.figure()
        ax = df1.plot(x = xaxis, y = feature, color = 'b', kind = 'scatter', title = feature)
        df2.plot(x = xaxis, y = feature, color = 'r', kind = 'scatter', ax = ax)
    

if __name__ == '__main__':
    main()