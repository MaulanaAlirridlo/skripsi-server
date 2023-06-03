from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import svm

def kFoldProcess(k, X, y):
  # clf = svm.SVC(kernel='rbf')
  clf = svm.SVC(kernel='poly')

  kf = KFold(n_splits=k)

  xTrain = []
  yTrain = []
  xTest = []
  yTest = []
  y_predict = []
  accuracies = []

  # Melakukan K-Fold Cross Validation
  for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Melatih model SVM pada data pelatihan
    clf.fit(X_train, y_train)

    # Memprediksi label pada data pengujian
    y_pred = clf.predict(X_test)

    # Menghitung akurasi
    accuracy = accuracy_score(y_test, y_pred)

    # Menyimpan data pada setiap fold
    xTrain.append(X_train)
    yTrain.append(y_train)
    xTest.append(X_test)
    yTest.append(y_test)
    y_predict.append(y_pred)
    accuracies.append(accuracy)

  # Menghitung nilai rata-rata akurasi
  mean_accuracy = sum(accuracies) / len(accuracies)

  return xTrain, yTrain, xTest, yTest, y_predict, accuracies, mean_accuracy  
