from sklearn import svm

def svmProcess(data, label, array):
    clf = svm.SVC()
    clf.fit(data, label)
    return clf.predict(array)