import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist,squareform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

set1 = np.genfromtxt('retreat2022/set1.dat',delimiter=',')
set2 = np.genfromtxt('retreat2022/set2.dat',delimiter=',')

def get_train_test(dataset):
    return train_test_split(dataset[:,0:4],dataset[:,-1], test_size=0.2, random_state=24)

def error(dataset,  model):
    model_name = type(model).__name__
    Xtrn, Xtst, ytrn, ytst = get_train_test(dataset)
    if model_name == 'KNeighborsClassifier':
        parameters = {'n_neighbors':[1, 20]}
        clf = GridSearchCV(model, parameters, scoring='accuracy')
        clf.fit(Xtrn, ytrn)
        return accuracy_score(ytst, clf.predict(Xtst))
    else:
        model.fit(Xtrn,ytrn)
        return accuracy_score(ytst, model.predict(Xtst))

def error2(dataset,  model):
    model_name = type(model).__name__
    Xtrn, Xtst, ytrn, ytst = get_train_test(dataset)
    if model_name == 'KNeighborsClassifier':
        parameters = {'n_neighbors':np.arange(1, 50)}
    elif model_name == 'LinearDiscriminantAnalysis':
        parameters = {'shrinkage':np.linspace(0.1,1,10)}
    clf = GridSearchCV(model, parameters, scoring='accuracy')
    clf.fit(Xtrn, ytrn)
    return accuracy_score(ytst, clf.predict(Xtst))


def main(dataset, model):
    return error(dataset, model)


print(main(set1,KNeighborsClassifier()))
print(main(set2,KNeighborsClassifier()))

print(main(set1,LinearDiscriminantAnalysis()))
print(main(set2, LinearDiscriminantAnalysis()))





