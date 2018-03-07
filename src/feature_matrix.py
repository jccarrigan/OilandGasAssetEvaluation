import functools as func
import datetime as dt
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA

def merge_and_fill(clean, categoricals=False):

    #join files together
    merge_1 = func.reduce(lambda x,y: pd.merge(x,y, how='left' ,on='API10'),clean[1:-1])
    merge_2 = pd.merge(merge_1, clean[7],how='left',on='PROPNUM')

    merge_2.drop('Unknown',axis=1,inplace=True)

    #Fill in null values
    features = merge_2.fillna(value=merge_2.median())
    features = features.fillna(value = {'PROJECT': 'Unknown','AdjustableFlag':'N','ProducingMethod': 'Flowing'})

    if categoricals:
        return features

    else:
        return features.drop(['LEASE','OPERATOR','RESERVOIR','COUNTY','FIELD','PROJECT','AdjustableFlag','ProducingMethod'],axis=1)

def simple_features(features):

    features  = features.copy()

    return features[['PROPNUM','API10','SH_LONGITUDE','TtlProppantLbs','BaseWaterGal','SH_LATITUDE','MD','24HourOil','OOIP','TestGas','LOWER_PERF','LATERAL_LENGTH','Porosity','Biocides']]

def merge_cat(features):

    #Grab categorical features
    api = features['API10'].copy()
    categorical = features[['LEASE','OPERATOR','RESERVOIR','COUNTY','FIELD','PROJECT','AdjustableFlag','ProducingMethod']].copy()

    #Grab numeric features
    numeric = features.drop(['LEASE','OPERATOR','RESERVOIR','COUNTY','FIELD','PROJECT','AdjustableFlag','ProducingMethod'],axis = 1)
    pca = PCA(n_components=500)
    coef = pca.fit_transform(pd.get_dummies(categorical))
    list1 = ['pc'+str(i) for i in range(1,501)]
    coefficients = pd.DataFrame(coef, columns =list1)
    coefficients['API10'] = api
    features_with_pca = pd.merge(numeric,coefficients, how='left',on='API10')

    return features_with_pca

def get_metric(production, years):

    production = production.copy()
    production = production.sort_values(['PROPNUM','DATE'])
    production.reset_index(inplace=True,drop=True)

    #Get wells that have on average more than 20 days on
    mean = production.groupby('PROPNUM').mean()
    active_wells = mean[mean['DAYSON']>21].index

    active_production = production[production['PROPNUM'].isin(active_wells)]
    active_production.reset_index(inplace=True,drop=True)

    #Get wells that have less than 36 months producing
    count = active_production.groupby('PROPNUM').count()
    new_wells = count[count['DATE']<(years+1)*12].index
    new_production = active_production[active_production['PROPNUM'].isin(new_wells)]

    #Get all data from this group after Jan 1 2015
    recent_production = new_production[new_production['DATE']>(pd.to_datetime('2017-01-01')-dt.timedelta(days=365*years))]

    #Get wells that have more than 17 months producing
    count = recent_production.groupby('PROPNUM').count()
    first_wells = count[count['DATE']>int(.75*years*12)].index
    first_production = recent_production[recent_production['PROPNUM'].isin(first_wells)]

    #Grab the first 24 months of those
    first = first_production.groupby('PROPNUM').head(years*12)

    counts = first.groupby(['PROPNUM','OIL']).size().to_frame('size')
    counts.reset_index(inplace=True)

    #Find the amount of 0 producing months
    zero = counts[counts['OIL'] == 0]

    #Get wells with more than 2 months not flowing
    non_flowing = zero[zero['size'] > 2*years]
    non_flowing.reset_index(inplace=True,drop=True)

    #Get wells who have 2 or less months not flowing
    flowing = first[first['PROPNUM'].isin(non_flowing['PROPNUM'])==False]
    flowing.reset_index(inplace=True,drop=True)

    #Grab the first 18 months from those
    output = flowing.groupby('PROPNUM').head(int((years-.5)*12))
    output.reset_index(inplace= True)

    output = output.copy()
    output.drop(['DAYSON','DATE'], axis=1,inplace=True)

    #Get metrics
    output_mean = output.groupby('PROPNUM').mean()
    output_cum = output.groupby('PROPNUM').sum()
    output_peak = output.groupby('PROPNUM').max()

    output_mean.reset_index(inplace=True)
    output_cum.reset_index(inplace=True)
    output_peak.reset_index(inplace=True)

    output_mean.drop('index', axis=1,inplace=True)
    output_cum.drop('index', axis=1,inplace=True)
    output_peak.drop('index', axis=1,inplace=True)

    return len(output_mean), [output_mean, output_cum,output_peak]

def merge_production(features,production_target):

    #Merge feature matrix and targets and fill nulls
    feature_matrix = pd.merge(features,production_target, on='PROPNUM',how='inner')
    feature_matrix = feature_matrix.fillna(feature_matrix.median())

    #Condense duplicate wells
    feature_matrix.drop('PROPNUM',axis=1, inplace = True)
    feature_matrix = feature_matrix.groupby('API10').mean()
    feature_matrix.reset_index(inplace=True)

    #Split metrics
    oil_metric = feature_matrix['OIL']
    gas_metric = feature_matrix['GAS']
    water_metric = feature_matrix['WATER']

    #Split features
    feature_matrix.drop(['OIL','GAS','WATER'],axis=1, inplace=True)

    return feature_matrix, [oil_metric,gas_metric,water_metric]

if __name__ == '__main__':

    #Load data
    with open('Data/clean.pkl','rb') as p:
        clean = pickle.load(p)

    #Grab production metrics
    production = clean[0].copy()
    years = 10
    well_count,production_metric = get_metric(production,years)
    metric_info = (years,well_count)

    #Grab features
    categoricals = False
    simple_model = True

    features = merge_and_fill(clean, categoricals)

    if simple_model:
        features = simple_features(features)

    if categoricals:
        features = merge_cat(features)

    #Merge and split features and metrics
    feature_matrix = []
    targets = []
    for i in range(3):
        f , t = merge_production(features,production_metric[i])
        feature_matrix.append(f)
        targets.append(t)

    with open('Data/features.pkl','wb') as p:
        pickle.dump(feature_matrix,p)

    with open('Data/targets.pkl','wb') as p:
        pickle.dump(targets,p)

    with open('Data/metricinfo.pkl','wb') as p:
        pickle.dump(metric_info,p)
