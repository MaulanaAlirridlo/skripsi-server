from sklearn import svm

def svmProcess(data, label, array):
    # clf = svm.SVC(kernel='rbf')
    clf = svm.SVC(kernel='poly')
    clf.fit(data, label)
    return clf.predict(array)