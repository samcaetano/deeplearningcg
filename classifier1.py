from sklearn import tree

features = [[1.9, 50, 1., 0.], [2.0, 55, 1., 0.],
            [1.5, 70, 1., 0.6], [1.3, 80, 0.2, 1],
            [1.0, 50, 0.3, 0.], [1.3, 45, 0.4, 0.1],
            [1.77, 68, 1., 1.], [1.8, 83, 0.4, 0.1]]
labels = ['elfo', 'elfo',
          'anao', 'anao',
          'hobbit', 'hobbit',
          'humano', 'humano']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels) # encontra os padroes

print clf.predict([[177, 64, 0.3, 0.]])
