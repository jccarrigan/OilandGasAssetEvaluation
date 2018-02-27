import pandas as pd
import numpy as np
import pickle

def load_files():

    #Importing CSV's
    production = pd.read_csv('Data/AriesProduction.csv')
    Wells = pd.read_csv('Data/AriesWells.csv',low_memory=False)
    FFru = pd.read_csv('Data/FracFocusRollUp.csv')
    geo = pd.read_csv('Data/WellGeo.csv')

    #Tables to import
    files = ['ND_FDMTRF.txt', 'ND_FM.txt', 'ND_PFT.txt', 'ND_PT.txt']

    #Importing txt files
    FDMTRF = pd.read_table('Data/'+files[0])
    FM = pd.read_table('Data/'+files[1])
    PFT = pd.read_table('Data/'+files[2])
    PT = pd.read_table('Data/'+files[3])

    #Storing data in list
    data = [production, Wells, FFru, geo, FDMTRF, FM, PFT, PT]

    return data

def cleanup(data):

    #Production information
    production_clean= data[0].copy()
    production_clean.drop(['GAS_SALES','GAS_VENTED'],axis=1,inplace=True)
    production_clean['DATE'] = pd.to_datetime(production_clean['P_DATE'])
    production_clean.drop('P_DATE',axis=1,inplace=True)

    #Well_information
    Wells_clean = data[1].copy()

    Wells_clean = Wells_clean[['PROPNUM','API_10','LEASE','OPERATOR','RESERVOIR','COUNTY','FIELD','PROJECT','UPPER_PERF','LOWER_PERF','LATERAL_LENGTH','MD','TVD','SH_LATITUDE','SH_LONGITUDE','BH_LATITUDE','BH_LONGITUDE']]

    Wells_clean = Wells_clean.rename(columns={'API_10':"API10"})

    #FracFocus information
    FFru_clean = data[2].copy()
    for items in FFru_clean.columns:
        if FFru_clean[items].isnull().sum() > 8000:
            FFru_clean.drop(items, axis=1,inplace=True)

    FFru_clean['API10'] = (FFru_clean['API']/10000).astype(int)
    FFru_clean.drop(['JobStartDate','JobEndDate','DisclosureId','API'], axis=1,inplace=True)

    #Geology
    geo = data[3].copy()
    geo.drop(['Unnamed: 0','unit'], inplace=True, axis=1)

    geo['OOIP'] = np.where(geo['parameter'] == 'OOIP',geo['value'],0)
    geo['Porosity'] = np.where(geo['parameter'] == 'POROSITY',geo['value'],0)

    OOIP_sum = geo.groupby('PROPNUM').sum()
    Porosity_mean = geo.groupby('PROPNUM').mean()

    OOIP_sum.drop(['value','Porosity'],axis=1,inplace=True)
    Porosity_mean.drop(['value','OOIP'],axis=1,inplace=True)

    OOIP_sum.reset_index(inplace=True)
    Porosity_mean.reset_index(inplace=True)

    geo_clean = pd.merge(OOIP_sum,Porosity_mean, on='PROPNUM',how='inner')

    #Completion information #1
    FDMTRF_clean = data[4].copy()
    for items in FDMTRF_clean.columns:
        if FDMTRF_clean[items].isnull().sum() > 5000:
            FDMTRF_clean.drop(items, axis=1,inplace=True)
    FDMTRF_clean.drop(['StateAbbreviation','API12'],axis=1,inplace=True)

    #Formation information
    FM_clean = data[5].copy()
    num_formations = FM_clean[['API10','formationName']].groupby('API10').count().reset_index()
    num_formations['num_formations'] = num_formations['formationName']
    num_formations.drop(['formationName'],axis=1,inplace = True)
    FM_clean = num_formations

    #Completion information #2
    PFT_clean = data[6].copy()
    PFT_clean.drop(['API12','StateAbbreviation','CompletionID','TreatmentWellbore','TreatmentAcidPercentage','TreatmentType','TreatmentDate'], axis=1,inplace=True)
    PFT_clean['JobThickness'] = PFT_clean['TreatmentBottom'] - PFT_clean['TreatmentTop']

    PFT_max = PFT_clean[['API10','maxTreatmentPressure','TreatmentMaxRate']].groupby('API10').max()
    PFT_max.reset_index(inplace=True)

    PFT_sum = PFT_clean[['API10','JobThickness']].groupby('API10').sum()
    PFT_sum.reset_index(inplace=True)

    PFT_clean = pd.merge(PFT_sum,PFT_max,on='API10')

    #Production test information
    PT_clean = data[7].copy()
    PT_clean.drop(['StateAbbreviation','CompletionID','testDate','SizeAndTypeOfPump','API12','ProductionGasDisposition'], axis=1,inplace=True)

    clean = [production_clean, Wells_clean, FFru_clean, FDMTRF_clean, FM_clean, PFT_clean, PT_clean,geo_clean]

    return clean

if __name__ == '__main__':

    #Import, clean and save data 
    clean = cleanup(load_files())
    with open('Data/clean.pkl','wb') as p:
        pickle.dump(clean,p)
