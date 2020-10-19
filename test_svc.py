import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))
clf.fit(X, y)
#Pipeline(steps=[('standardscaler', StandardScaler()),('svc', SVC(gamma='auto'))])
print(clf.predict([[-0.8, -1]]))
print(clf.predict_proba([[-0.8, -1]]))