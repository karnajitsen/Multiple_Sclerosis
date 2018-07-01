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
    dfUSControlTest = pd.DataFrame(pd.pivot_table(dfUSControl, index = "floodlightOpenId", columns = "testMetricName", values = "testResultMetricValue").to_records())
    dfUSTreatedTest = pd.DataFrame(pd.pivot_table(dfUSTreated, index = "floodlightOpenId", columns = "testMetricName", values = "testResultMetricValue").to_records())
    dfUSControlTest.fillna(0, inplace = True)
    dfUSTreatedTest.fillna(0, inplace = True)

if __name__ == '__main__':
    main()