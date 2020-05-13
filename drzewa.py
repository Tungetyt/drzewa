import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from plotkab import plot_decision_regions
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
import pydot



    

def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # standardyzacja cech
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))


    tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    tree.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    a1 = tree.score(X_train, y_train)
    print(f'Gini, Accuracy:  {a1}')
    plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.savefig('tree4')
    plt.show()
    
    export_graphviz(tree,out_file='drzewo.dot', feature_names=['Długość płatka', 'Szerokość płatka'])
    (graph,) = pydot.graph_from_dot_file('drzewo.dot')
    graph.write_png('drzewo1.png');
    
 
    
    tree_e = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)
    tree_e.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    a2 = tree_e.score(X_train, y_train)
    print(f'Entropy, Accuracy:  {a2}')
    plot_decision_regions(X_combined, y_combined, classifier=tree_e, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.savefig('tree4_e')
    plt.show()
    
    export_graphviz(tree_e,out_file='drzewo_e.dot', feature_names=['Długość płatka', 'Szerokość płatka'])
    (graph2,) = pydot.graph_from_dot_file('drzewo_e.dot')
    graph2.write_png('drzewo1_e.png')
    
    tree2 = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=1)
    tree2.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    a3 = tree2.score(X_train, y_train)
    print(f'Gini, Accuracy:  {a3}')
    plot_decision_regions(X_combined, y_combined, classifier=tree2, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.savefig('tree8')
    plt.show()
    
    export_graphviz(tree2,out_file='drzewo2.dot', feature_names=['Długość płatka', 'Szerokość płatka'])
    (graph3,) = pydot.graph_from_dot_file('drzewo2.dot')
    graph3.write_png('drzewo2.png')
    
    tree2_e = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=1)
    tree2_e.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    a4 = tree2_e.score(X_train, y_train)
    print(f'Entropy, Accuracy:  {a4}')
    plot_decision_regions(X_combined, y_combined, classifier=tree2_e, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.savefig('tree8_e')
    plt.show()
    
    export_graphviz(tree2_e,out_file='drzewo2_e.dot', feature_names=['Długość płatka', 'Szerokość płatka'])
    (graph4,) = pydot.graph_from_dot_file('drzewo2_e.dot')
    graph4.write_png('drzewo2_e.png')

    forest = RandomForestClassifier(criterion='gini', n_estimators=15, random_state=1, n_jobs=2)
    forest.fit(X_train, y_train)
    #pred = forest.predict(X_test) #prediction
    f = forest.score(X_train,y_train)
    plot_decision_regions(X_combined, y_combined,
    classifier=forest, test_idx=range(105,150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.savefig('randomforest')
    plt.show()
    print(f'Gini, forrest Accuracy:  {f}')


    forest2 = RandomForestClassifier(criterion='gini', n_estimators=20, random_state=1, n_jobs=2)
    forest2.fit(X_train, y_train)
   
    f2 = forest2.score(X_train,y_train)
    plot_decision_regions(X_combined, y_combined,
    classifier=forest2, test_idx=range(105,150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.savefig('randomforest')
    plt.show()
    print(f'Gini, forrest Accuracy:  {f2}')
 


if __name__ == '__main__':
    main()
