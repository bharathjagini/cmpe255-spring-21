import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        self.X_test = None
        self.y_test = None
        

    def define_feature(self,feature):
        #feature_cols = ['pregnant', 'glucose', 'bp', 'bmi','pedigree']
        feature_cols =feature
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y
    
    def train(self,feature):
        # split X and y into training and testing sets
        X, y = self.define_feature(feature)
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=0)
        # train a logistic regression model on the training set
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        return logreg
    
    def predict(self,feature):
        model = self.train(feature)
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)
    
if __name__ == "__main__":
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    classifer = DiabetesClassifier()
    #BaseLine Feature
    output=['Experiment','Accuracy','Confusion Matrix','Comment']
    output_df = pd.DataFrame(columns=output)
    baseLineFeature=['pregnant', 'insulin', 'bmi', 'age']
    result = classifer.predict(baseLineFeature)
    score = classifer.calculate_accuracy(result)
    con_matrix = classifer.confusion_matrix(result)
    baseLineResult=['BaseLine',score,con_matrix,'BaseLine Features '+str(baseLineFeature)]
    output_df=output_df.append(pd.Series(baseLineResult,output),ignore_index=True)
    #Other Features
    feature=[['pregnant', 'glucose', 'bp', 'bmi'],['pregnant', 'glucose', 'bp', 'bmi','pedigree'],
    ['pregnant', 'glucose', 'bp', 'bmi','age'],
            ]
    
    x=0
    for feat in feature:
     result = classifer.predict(feat)
     score = classifer.calculate_accuracy(result)
     con_matrix = classifer.confusion_matrix(result)
     x=x+1
     newRow=['Solution '+str(x),score,con_matrix,'Features used'+str(feat)]
     output_df=output_df.append(pd.Series(newRow,output),ignore_index=True)
    print(output_df)
    
