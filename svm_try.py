from sklearn.metrics import confusion_matrix
from sklearn import svm

clf = svm.SVC(C=50, gamma=0.015, verbose=True)
clf.fit(train_images, train_labels)
print(clf.score(validation_images, validation_labels))
#y_pred = clf.predict(test_images)
v_pred = clf.predict(validation_images)
