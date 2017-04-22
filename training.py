import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])	

def oversampling(x_train,y_train,c):
	m = x_train.shape[0]
	pos = x_train[0].reshape(1,-1)
	ones = y_train[0]
	for i in range(0,m):
		if y_train[i] == 1:
			pos = np.append(pos,x_train[i].reshape(1,-1),axis=0)
			ones = np.append(ones,np.array([1]))
	for i in range(0,c):
		pos = np.append(pos,pos,axis=0)
		ones = np.append(ones,ones)
	return np.append(x_train,pos,axis=0),np.append(y_train,ones)
	
x_train = load_sparse_csr("onehotencoding300.npz")
y_train = np.load("train_y300.npz")
y_train = y_train.astype(int)
print x_train.shape
print y_train[0] == 0
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

model = LogisticRegression(C=1.2)
kfold = KFold(n_splits=10, random_state=7)
scoring = 'roc_auc'
results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
print("auc: %.3f (%.3f)") % (results.mean(), results.std())

model.fit(X_train,y_train)
print confusion_matrix(y_test,model.predict(X_test))

model = SVM()
kfold = KFold(n_splits=10, random_state=7)
scoring = 'roc_auc'
results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
print("auc: %.3f (%.3f)") % (results.mean(), results.std())

model.fit(X_train,y_train)
print confusion_matrix(y_test,model.predict(X_test))

model = LogisticRegression(class_weight='balanced',C=1.2)
kfold = KFold(n_splits=2, random_state=7)
scoring = 'roc_auc'
results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
print("auc: %.3f (%.3f)") % (results.mean(), results.std())

model.fit(X_train,y_train)
print confusion_matrix(y_test,model.predict(X_test))

model = SVM(class_weight='balanced')
kfold = KFold(n_splits=10, random_state=7)
scoring = 'roc_auc'
results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
print("auc: %.3f (%.3f)") % (results.mean(), results.std())

model.fit(X_train,y_train)
print confusion_matrix(y_test,model.predict(X_test))

x_train = np.load("train_Perc.npy")
y_train = np.load("train_y300.npz")
y_train = y_train.astype(int)
print x_train.shape
print y_train[0] == 0
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

X_train,y_train = oversampling(X_train,y_train,5)
model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.5,max_depth=1, random_state=0).fit(X_train, y_train)
print confusion_matrix(y_test,model.predict(X_test))
