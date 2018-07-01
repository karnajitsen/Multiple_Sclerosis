import pandas as pd
import numpy as np
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)



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
    
    tmn = dfUS.testMetricName.unique()
    dfUSControl = dfUS[dfUS.participantIsControl == True].copy()
    dfUSTreated = dfUS[dfUS.participantIsControl == False].copy()
    dfUSControl.drop(columns = ['testResultMetricId', 'testResultMetricCreatedOn'], inplace = True)
    dfUSTreated.drop(columns = ['testResultMetricId', 'testResultMetricCreatedOn'], inplace =True)
    dfUSControl.drop_duplicates(inplace = True)
    dfUSTreated.drop_duplicates(inplace = True)
    dfUSControlTest = pd.DataFrame(data = dfUSControl.floodlightOpenId.unique(), columns = {"floodlightOpenId"})
    dfUSTreatedTest =  pd.DataFrame(data = dfUSTreated.floodlightOpenId.unique(), columns = {"floodlightOpenId"})
    d = pd.DataFrame()
    d1 = pd.DataFrame()
    for t in tmn:
        d = pd.DataFrame()
        d1 = pd.DataFrame()
        d[[ "floodlightOpenId" , t]] = dfUSTreated[dfUSTreated.testMetricName == t][["floodlightOpenId", "testResultMetricValue"]]
        #d.dropna(inplace = True)
        dfUSTreatedTest = pd.merge(dfUSTreatedTest, d, how = "left", on = "floodlightOpenId")
        #d1[[ "floodlightOpenId" , t]]= dfUSControl[dfUSControl.testMetricName == t][["floodlightOpenId", "testResultMetricValue"]]
        #dfUSControlTest = pd.concat([dfUSControlTest, d1], axis = 1)
    
    
    

if __name__ == '__main__':
    main()