import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        #print(string_columns)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')
        #print(self.df)

    def validate(self):
      np.random.seed(2)
      n = len(self.df)
      n_val = int(0.2 * n)
      n_test = int(0.2 * n)
      n_train = n - (n_val + n_test)
      #print(n,n_val,n_test,n_train)
      idx = np.arange(n)
      np.random.shuffle(idx)
      df_shuffled = self.df.iloc[idx]
      df_train = df_shuffled.iloc[:n_train].copy()
      df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
      df_test = df_shuffled.iloc[n_train+n_val:].copy()

      y_train_orig = df_train.msrp.values
      y_val_orig = df_val.msrp.values
      y_test_orig = df_test.msrp.values
      y_train = np.log1p(df_train.msrp.values)
      y_val = np.log1p(df_val.msrp.values)
      y_test = np.log1p(df_test.msrp.values)

      del df_train['msrp']
      del df_val['msrp']
      del df_test['msrp']
      return y_train,df_train,y_val,df_val,y_val_orig,y_test,df_test,y_test_orig
      
    def linear_regression(self, X, y,r=1):
         ones = np.ones(X.shape[0])
         X = np.column_stack([ones, X])
         XTX = X.T.dot(X)
         XTX = XTX+(r*np.eye(XTX.shape[0]))
         XTX_inv = np.linalg.inv(XTX)
         w = XTX_inv.dot(X.T).dot(y)  
         return w[0], w[1:]
    
    def prepare_X(self,base,df_train):
      df_num = df_train[base]
      #print(df_num)
      df_num = df_num.fillna(0)
      X = df_num.values
      return X
    def rmse(self,y, y_pred):
     error = y_pred - y
     mse = (error ** 2).mean()
     return np.sqrt(mse)

    def getPredictedPrices(self,df_val,y_val_orig,y_val_pred):
       df_val['msrp']=y_val_orig 
       df_val['msrp_pred']=np.expm1(y_val_pred)       
       print(df_val[['engine_cylinders','transmission_type','driven_wheels','number_of_doors','market_category','vehicle_size','vehicle_style','highway_mpg','city_mpg','popularity','msrp','msrp_pred']].head())
      
if __name__ == "__main__":
    pd.set_option('display.max_columns', 15)
    carprice= CarPrice()
    carprice.trim()
    y_train,df_train,y_val,df_val,y_val_orig,y_test,df_test,y_test_orig=carprice.validate()

    base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
    X_train =carprice.prepare_X(base,df_train)
    
    for r in [0.00001,0.0001,0.001,0.01,0.1,1,10]:
     w_0, w = carprice.linear_regression(X_train, y_train,r)
     y_pred = w_0 + X_train.dot(w)
     X_val = carprice.prepare_X(base,df_val)
     y_val_pred = w_0 + X_val.dot(w)
     print(r,carprice.rmse(y_val,y_val_pred))
   
    # Got minimum rmse at r = 1 for validation data
    w_0, w = carprice.linear_regression(X_train, y_train) 
    #Test dataset
    X_test=carprice.prepare_X(base,df_test)
    y_test_pred=w_0 + X_test.dot(w)
   
   
    carprice.getPredictedPrices(df_test,y_test_orig,y_test_pred)