import os
from matplotlib import pyplot as plt
from sklearn import tree
import graphviz
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


def glass():
    df = pd.read_csv('glass.csv', sep=',')
    df = df.drop('Id', 1)
    data = df.drop('Type', 1).values
    target = df['Type'].values
    le = preprocessing.LabelEncoder()
    targets_encoded = le.fit_transform(target)
    for i in range(1, 20):
        clf = tree.DecisionTreeClassifier(max_depth=i, max_leaf_nodes=i+1, splitter='random')
        clf = clf.fit(data, targets_encoded)
        dot_data = tree.export_graphviz(clf, out_file=None,
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(os.getcwd() + '\\glasses\\Glass{0}'.format(i))
        print('accuracy: {0}; max_depth = {1}'.format(metrics.accuracy_score(targets_encoded, clf.predict(data)), i))
    print('******')
    for i in range(1, 20):
        clf = tree.DecisionTreeClassifier(max_depth=i, max_leaf_nodes=i+1, criterion='entropy')
        clf = clf.fit(data, targets_encoded)
        dot_data = tree.export_graphviz(clf, out_file=None,
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(os.getcwd() + '\\glasses1\\Glass{0}'.format(i))
        print('accuracy: {0}; max_depth = {1}'.format(metrics.accuracy_score(targets_encoded, clf.predict(data)), i))



def lenses():
    clf = tree.DecisionTreeClassifier()
    data = []
    target = []
    with open('Lenses.txt', 'r') as inp_f:
        for i in inp_f:
            data.append(i.split('  ')[1:5])
            target.append(i[-2::].strip())
    le = preprocessing.LabelEncoder()
    data_enc = [le.fit_transform(i) for i in data]
    target_enc = le.fit_transform(target)
    clf = clf.fit(data_enc, target_enc)
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render('Lenses')
    example = [[2, 1, 1, 1]]
    example_enc = [le.fit_transform(i) for i in example]
    reslt = clf.predict(example_enc)[0] + 1
    print('Predicted_type: {0}'.format(reslt))
    print('accuracy: {0}'.format(metrics.accuracy_score(target_enc, clf.predict(data_enc))))


def spam7():
    df = pd.read_csv('spam7.csv', sep=',')
    data = df.drop('yesno', 1).values
    target = df['yesno'].values
    le = preprocessing.LabelEncoder()
    target_enc = le.fit_transform(target)
    #parameters = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random'), 'max_depth': [1, 10]}
    #tree_clf = tree.DecisionTreeClassifier()
    #clf = GridSearchCV(tree_clf, parameters, cv=5)
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=1, min_samples_split=2, splitter='best')
    clf.fit(data, target_enc)
    #print(clf.best_estimator)
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    filled=True, rounded=True,
                                   special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render('spam7')



os.environ["PATH"] += os.pathsep + os.getcwd() + '\\release\\bin\\'
glass()
lenses()
spam7()