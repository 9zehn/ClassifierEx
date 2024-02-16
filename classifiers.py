from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
# A selection of classifiers and fine-tuning of parameters
# @Author: Leon Gruber
# @Date: September 2023
"""

class Classifiers():

    def __init__(self,data):
        # Converting pandas dataframe and splitting into training and testing data
        # Preparing data

        X = data[['A','B']].values
        y = data['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

        self.training_data = X_train
        self.training_labels = y_train
        self.testing_data = X_test
        self.testing_labels = y_test
        self.outputs = []


    def test_clf(self, clf, classifier_name=''):
        # Fitting the classifier and extrach the best score, training score and parameters

        clf.fit(self.training_data,self.training_labels)

        best_model = clf.best_estimator_

        y_predictions = best_model.predict(self.testing_data)
        accuracy_score_test = accuracy_score(self.testing_labels,y_predictions)

        self.plot(self.testing_data,y_predictions,model=clf,classifier_name=classifier_name)

        self.outputs.append(classifier_name + ", {:.3f}, {:.3f}".format(clf.best_score_, accuracy_score_test))

        return (clf.best_score_,clf.best_params_)


    def classifyNearestNeighbors(self):
        # Nearest Neighbors classifier

        parameters = {'n_neighbors': list(range(1,20)),
                      'leaf_size':   [5*n for n in range(1,7)]}

        model = KNeighborsClassifier()
        grid_search = GridSearchCV(model,parameters,cv=5)

        best_score, best_params = self.test_clf(grid_search,classifier_name="kNearestNeighbors")

        print("Best score: ",best_score)
        print("Best parameters: ",best_params)


    def classifyLogisticRegression(self):
        # Logistic Regression classifier

        parameters = {'C': [0.1,0.5,1,5,10,50,100]}

        model = LogisticRegression()
        grid_search = GridSearchCV(model,parameters,cv=5)

        best_score, best_params = self.test_clf(grid_search,classifier_name="LogisticRegression")

        print("Best score: ",best_score)
        print("Best parameters: ",best_params)


    def classifyDecisionTree(self):
        # Decision Tree classifier

        parameters = {'max_depth': list(range(1,51)),
                      'min_samples_split': list(range(2,11))}

        model = DecisionTreeClassifier()
        grid_search = GridSearchCV(model,parameters,cv=5)

        best_score, best_params = self.test_clf(grid_search,classifier_name="DecisionTree")

        print("Best score: ",best_score)
        print("Best parameters: ",best_params)



    def classifyRandomForest(self):
        # Random Forest classifier

        parameters = {'max_depth': list(range(1,6)),
                      'min_samples_split': list(range(2,11))}

        model = RandomForestClassifier()
        grid_search = GridSearchCV(model,parameters,cv=5)

        best_score, best_params = self.test_clf(grid_search,classifier_name="RandomForest")

        print("Best score: ",best_score)
        print("Best parameters: ",best_params)


    def classifyAdaBoost(self):
        # AdaBoost classifier
        parameters = {'n_estimators': [n*10 for n in range(1,8)],
                      'estimator__max_depth': [1,2,3,4,5]}

        estimator = DecisionTreeClassifier()

        model = AdaBoostClassifier(estimator=estimator)
        grid_search = GridSearchCV(model,parameters,cv=5)

        best_score, best_params = self.test_clf(grid_search,classifier_name="AdaBoost")

        print("Best score: ",best_score)
        print("Best parameters: ",best_params)


    def plot(self, X, Y, model,classifier_name = ''):
        X1 = X[:, 0]
        X2 = X[:, 1]

        X1_min, X1_max = min(X1) - 0.5, max(X1) + 0.5
        X2_min, X2_max = min(X2) - 0.5, max(X2) + 0.5

        X1_inc = (X1_max - X1_min) / 200.
        X2_inc = (X2_max - X2_min) / 200.

        X1_surf = np.arange(X1_min, X1_max, X1_inc)
        X2_surf = np.arange(X2_min, X2_max, X2_inc)
        X1_surf, X2_surf = np.meshgrid(X1_surf, X2_surf)

        L_surf = model.predict(np.c_[X1_surf.ravel(), X2_surf.ravel()])
        L_surf = L_surf.reshape(X1_surf.shape)

        plt.title(classifier_name)
        plt.contourf(X1_surf, X2_surf, L_surf, cmap = plt.cm.coolwarm, zorder = 1)
        plt.scatter(X1, X2, s = 38, c = Y)

        plt.margins(0.0)
        # uncomment the following line to save images
        plt.savefig(f'{classifier_name}.png')
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv('input.csv')
    models = Classifiers(df)

    # Plot:
    X1 = df['A']
    X2 = df['B']
    Y = df['label']
    plt.title("Original Data")
    plt.scatter(X1,X2,c=Y)
    plt.show()


    print('Classifying with NN...')
    models.classifyNearestNeighbors()
    print('Classifying with Logistic Regression...')
    models.classifyLogisticRegression()
    print('Classifying with Decision Tree...')
    models.classifyDecisionTree()
    print('Classifying with Random Forest...')
    models.classifyRandomForest()
    print('Classifying with AdaBoost...')
    models.classifyAdaBoost()

    with open("output.csv", "w") as f:
        print('Name, Best Training Score, Testing Score',file=f)
        for line in models.outputs:
            print(line, file=f)
