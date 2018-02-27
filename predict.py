from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt


def final_model(feature_matrix, target):

    #Train/test split the data
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix,target,test_size=.2)

    #Instantiate Scaler and Model
    ss = StandardScaler()
    model = CatBoostRegressor()

    #Scale data
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.fit_transform(X_test)

    #Fit model and predict
    model.fit(X_train_scaled,y_train)
    y_pred = pd.Series(model.predict(X_test_scaled))

    #Validate model
    RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)

    print("RMSE (test data):{:.2f}".format(RMSE))
    print("r2 (test data):{:.2f}".format(r2))

    feature_importance = set()

    catboost = True
    tree = False
    if catboost:
        feature_importance = model.get_feature_importance(X_train_scaled,y=y_train)
    elif tree:
        feature_importance = model.feature_importances_


    return y_test, y_pred, RMSE,r2, feature_importance

def make_plots(y_test, y_pred,metric,timeframe):

    if metric == 'Cumulative':
        label = 'Oil Production, bbls'
    else:
        label = 'Oil Production, bbl/month'

    #Distribution comparison
    limit = int(max(y_pred))
    plt.figure(figsize=(10,10))
    sns.set_style('darkgrid')
    plt.xlim((0,limit))
    sns.distplot(y_test,bins=35, axlabel = metric + label,label='y_test', color='purple').set_title(str(timeframe)+ ' Year Production', fontsize=12)
    sns.distplot(y_pred,bins=35, axlabel = metric + label,label='y_pred',color='black').set_title(str(timeframe) + ' Year '+metric+ ' Production', fontsize=12)
    plt.yticks([])
    plt.ylabel('Frequency',fontsize=12)
    plt.legend(fontsize=12)
    plt.xlabel(metric + ' Oil Production, bbl/month',fontsize=12)

    plt.savefig('Plots/'+str(timeframe)+'Year/'+'DistributionComparison'+metric+'.png')

    #test/prediction correlation

    plt.figure(figsize=(10,10))
    sns.set_style('darkgrid')
    sns.regplot(y_test,y_pred, color='r',label='test vs prediction')
    plt.xlabel('y_test, ' + label)
    plt.ylabel('y_predict, ' + label)
    plt.xlim((0,limit))
    plt.ylim((0,limit))
    plt.plot(list(range(limit)),list(range(limit)), color='green',label='Perfect correlation')
    plt.legend()
    plt.title('Correlation for ' + str(timeframe) + ' Year ' + metric+ ' Oil Production')

    plt.savefig('Plots/'+str(timeframe)+'Year/'+'Correlation'+metric+'.png')


def feature_importance_plot(columns,feature_importance, top_amount, metric,timeframe):

    #Top features
    list1 = list(zip(columns,feature_importance))
    list1 = sorted(list1,key= lambda x: x[1], reverse=True)
    list2 = [x[0] for x in list1]
    list3 = [x[1] for x in list1]

    features = list2[:top_amount][::-1]
    importances = list3[:top_amount][::-1]
    indices = np.argsort(importances)

    plt.figure(figsize=(10,10))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)),importances, color='g', align='center')
    plt.yticks(range(len(indices)), features)
    plt.xlabel('Relative Importance')

    plt.savefig('Plots/'+str(timeframe)+'Year/'+'feature_importance'+metric+'.png')

if __name__ == '__main__':

    np.random.seed(50)

    #Load data
    with open('Data/features.pkl','rb') as p:
        features= pickle.load(p)

    with open('Data/targets.pkl','rb') as p:
        targets = pickle.load(p)

    with open('Data/metricinfo.pkl','rb') as p:
        metric_info = pickle.load(p)


    #Fit and Validate model
    API = []
    models = ['Average','Cumulative','Peak']
    results = {}

    for i in range(3):

        API.append(features[i].pop('API10'))

        y_test,y_pred,RMSE,r2,feature_importance = final_model(features[i],targets[i][0])

        results[models[i]] = {'RMSE':RMSE,'r2':r2}

        #Create distribution comparisons and top features plots
        make_plots(y_test, y_pred, metric=models[i], timeframe=metric_info[0])
        feature_importance_plot(features[i].columns, feature_importance, top_amount=10, metric=models[i], timeframe=metric_info[0])

    print (results.items())
    print ("Number of Wells:" + str(metric_info[1]))
