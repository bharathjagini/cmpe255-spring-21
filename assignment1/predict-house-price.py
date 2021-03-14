import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


class HousePrice:

    def __init__(self):
        boston_list=[ 'CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO','B','LSTAT', 'MEDV']
        self.df = pd.read_csv('data/housing.csv', header=None, delimiter=r"\s+", names=boston_list)
        print(f'${len(self.df)} lines loaded')

    def splitData(self):
       n=len(self.df)

       n_train = int(0.8 * n)
       n_test = int(0.2 * n)
       print(n,n_train,n_test)
       df_train = self.df.iloc[n_test:n_train].copy()
       df_test = self.df.iloc[:n_test].copy()
       return df_train,df_test

    def linearRegression(self,df_train,df_test):
       features=['LSTAT']
       
       X_train=df_train[features].copy()
       Y_train=df_train['MEDV'].copy()
       X_test=df_test[features].copy()
       Y_test=df_test['MEDV'].copy()
    
       lin_reg = LinearRegression()
       lin_reg.fit(X_train, Y_train)
    
       Y_train_pred=lin_reg.predict(X_train)
       
       print("Linear Regression")
       print("Training Data model performance")
       rmse_train = (np.sqrt(mean_squared_error(Y_train, Y_train_pred)))
    
      
       print('RMSE:',rmse_train)
       r2 = r2_score(Y_train, Y_train_pred)
       print('R2 score:',r2)
       Y_test_pred = lin_reg.predict(X_test)
       print("Test Data model performance")
       rmse_test = (np.sqrt(mean_squared_error(Y_test, Y_test_pred)))
       print('RMSE:',rmse_test)
       r2 = r2_score(Y_test, Y_test_pred)
       print('R2 score:',r2)
       
       plt.scatter(X_test.values.flatten(),Y_test)
       plt.plot(X_test.values.flatten(),Y_test_pred)
       plt.title("Linear regression")
       plt.xlabel("Datapoint")
       plt.ylabel("Predicted value")
       plt.show()
        
    
    def polynomialRegression(self,df_train,df_test,degree):
        features=['LSTAT']    
        X_train=df_train[features].copy()
        Y_train=df_train['MEDV'].copy()
        X_test=df_test[features].copy()
        Y_test=df_test['MEDV'].copy()
    
        poly_reg = PolynomialFeatures(degree=degree) 
        X_poly_train = poly_reg.fit_transform(X_train)
        lin_reg = LinearRegression() 
        lin_reg.fit(X_poly_train, Y_train) 
        Y_train_pred=lin_reg.predict(X_poly_train)
        print("Polynomial Regression")
        print("Training Data model performance")
        rmse_train = (np.sqrt(mean_squared_error(Y_train, Y_train_pred)))
        print('RMSE:',rmse_train)
        r2 = r2_score(Y_train, Y_train_pred)
        print('R2 score:',r2)
        
        X_poly_test = poly_reg.fit_transform(X_test)
        Y_test_pred=lin_reg.predict(X_poly_test)
        print("Test Data model performance")
        rmse_train = (np.sqrt(mean_squared_error(Y_test, Y_test_pred)))
        print('RMSE:',rmse_train)
        r2 = r2_score(Y_test, Y_test_pred)
        print('R2 score:',r2)
        
     
        plt.scatter(X_test.values.flatten(),Y_test)
        plt.plot(X_test.values.flatten(),Y_test_pred,"b-")
        plt.title("Polynomial regression of degree:{}".format(degree))
        plt.xlabel("Datapoint")
        plt.ylabel("Predicted value")
        plt.show()

        
        
    def multipleRegression(self,df_train,df_test):
        features=['LSTAT','RM','CRIM']        
        X_train=df_train[features].copy()
        Y_train=df_train['MEDV'].copy()
        X_test=df_test[features].copy()
        Y_test=df_test['MEDV'].copy()
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, Y_train)
    
        Y_train_pred=lin_reg.predict(X_train)
       
        print("Multiple Regression:::")
        print("Training Data model performance")
        rmse_train = (np.sqrt(mean_squared_error(Y_train, Y_train_pred)))    
        print('RMSE:',rmse_train)

        r2 = r2_score(Y_train, Y_train_pred)
        print('R2 score:',r2)
        print(len(X_train))
        
        r2_adj_train=1-(((1-r2)*(len(X_train)-1))/(len(X_train)-3-1))
        print('adjusted r2:',r2_adj_train)
        
        Y_test_pred=lin_reg.predict(X_test)
        print("Test Data model performance")
        rmse_test = (np.sqrt(mean_squared_error(Y_test, Y_test_pred)))
          
        print('RMSE:',rmse_test)
        r2 = r2_score(Y_test, Y_test_pred)
        print('R2 score:',r2)
        r2_adj_test=1-(((1-r2)*(len(X_test)-1))/(len(X_test)-3-1))
        print('adjusted r2:',r2_adj_test)
       
    
if __name__ == "__main__":
 housePrice=HousePrice()
 df_train,df_test=housePrice.splitData()
 housePrice.linearRegression(df_train,df_test)
 housePrice.polynomialRegression(df_train,df_test,2)
 housePrice.polynomialRegression(df_train,df_test,20)
 housePrice.multipleRegression(df_train,df_test)


