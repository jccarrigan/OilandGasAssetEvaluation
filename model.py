from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import numpy as np


def modeling(feature_matrix,target):

    #Train/test split the data
    X_train,X_test, y_train, y_test = train_test_split(feature_matrix.drop('API10',axis=1),target,test_size=.2)

    #Instantiate Scaler and Model
    ss = StandardScaler()
    model = CatBoostRegressor()

    #Scale data
    X_train_scaled = ss.fit_transform(X_train)

    #Validate Model
    RMSE = np.sqrt(-(cross_val_score(model,X_train_scaled,y_train,scoring='neg_mean_squared_error',cv=5).mean()))
    r2 = cross_val_score(model,X_train_scaled,y_train,scoring='r2',cv=5).mean()

    option = 1

    if option == 0:

        print("RMSE (two year production):{:.2f}".format(RMSE))
        print("r2 (two year production):{:.2f}".format(r2))

    elif option == 1:

        print("RMSE (five_year production):{:.2f}".format(RMSE))
        print("r2 (five year production):{:.2f}".format(r2))

    return RMSE,r2


def GridSearch(feature_matrix,targets):

    params = {'depth':[3,1,2,6,4,5,7,8,9,10],
      'iterations':[250,100,500,1000],
      'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3],
      'l2_leaf_reg':[3,1,5,10,100],
      'border_count':[32,5,10,20,50,100,200],
      'ctr_border_count':[50,5,10,20,100,200],
      'thread_count':4}

    #Grid Search for optimal parameters
    cb = CatBoostRegressor()

    model = GridSearchCV(cb,params,scoring='neg_mean_squared_error')
    model.fit(feature_matrix, targets[0])
    return model.best_params_

def final_model(feature_matrix, target):

    #Train/test split the data
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix.drop('API10',axis=1),target,test_size=.2)

    #Instantiate Scaler and Model
    ss = StandardScaler()
    model = CatBoostRegressor(border_count=100,l2_leaf_reg =1,iterations=1000, learning_rate=0.1,depth=9)

    #Scale data
    X_train_scaled = ss.fit_transform(X_train)

    #Validate model
    RMSE = np.sqrt(-(cross_val_score(model,X_train_scaled,y_train,scoring='neg_mean_squared_error',cv=5).mean()))
    r2 = cross_val_score(model,X_train_scaled,y_train,scoring='r2',cv=5).mean()

    return RMSE,r2

if __name__ == '__main__':

    np.random.seed(50)

    #Load data
    with open('Data/features.pkl','rb') as p:
        features= pickle.load(p)

    with open('Data/targets.pkl','rb') as p:
        targets = pickle.load(p)

    #Run modeling
    modeling(features,targets[0])s
