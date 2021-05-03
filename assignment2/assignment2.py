from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plotter
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns;

class SVMClassifier:
 def load_data(self):
    faces = fetch_lfw_people(min_faces_per_person=60)
    print('data loaded')
    print(faces.target_names)
    #print(faces.images)
    return faces

 def randomPCA(self,faces):
    pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)
    return model

 def splitData(self,faces):
    X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=2)
    return X_train,X_test,y_train,y_test
 
 def gridSearch(self,model,X_train,y_train):
    parameters = {'svc__C' : [1, 5, 10, 50], 
              'svc__gamma' : [0.0001, 0.0005, 0.001, 0.005]}
    clf = GridSearchCV(model, parameters)
    clf.fit(X_train, y_train)
    new_model = clf.best_estimator_
    print(clf.best_params_)
    return new_model

 def predict(self,new_model,X_test):
    new_labels = new_model.predict(X_test)
    print(new_labels)
    return new_labels

 def plot_results(self,data,faces, y_test,new_labels):
    fig, ax = plotter.subplots(3, 5)
    for i, axi in enumerate(ax.flat):
        axi.imshow(data[i].reshape(62,47), cmap='bone')
        axi.set(xticks=[], yticks=[])
        if y_test[i] == new_labels[i]:
            axi.set_ylabel(faces.target_names[new_labels[i]].split()[-1], color = "black")
        else:
            axi.set_ylabel(faces.target_names[new_labels[i]].split()[-1], color = "red")
        #axi.plot(i)
    plotter.show()


if __name__ == "__main__":
 classifier=SVMClassifier()
 faces=classifier.load_data()
 model=classifier.randomPCA(faces)
 X_train,X_test,y_train,y_test=classifier.splitData(faces)
 new_model=classifier.gridSearch(model,X_train,y_train)
 new_labels=classifier.predict(new_model, X_test)
 print(classification_report(y_test, new_labels,  target_names=faces.target_names))
 #print(confusion_matrix(y_test, new_labels, labels=range(faces.target_names.shape[0])))
 classifier.plot_results(X_test,faces,y_test,new_labels)
 confmatrix=confusion_matrix(y_test, new_labels, labels=range(faces.target_names.shape[0]))
 sns.heatmap(confmatrix,annot=True)
 plotter.title('Confusion matrix')
 plotter.xlabel('Actual label')
 plotter.ylabel('Predicted label')
 plotter.show()

