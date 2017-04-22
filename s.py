import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def loadArray(filename):
	a = np.load(filename+".npz")
	return a
	
def makeArray():
	with open('x_train.csv','rb') as infile:
		infile.next()
		A = [] #array for 300
		for line in infile:
			A.append(line.split(","))
	print len(A[0])
	return A

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
						 
def saveData(a,filename):
	f = open(filename+".npz",'wb')
	np.save(f,a)
		
def saveIntoChunks_csr(a,filename):
	for i in range(0,9):
		print i
		save_sparse_csr(filename+str(i)+".npz",a)
		
def makeFeatures(a,index,numberOfClasses,m):
	seen = np.array([0.0]*numberOfClasses)
	numberOfPos = 0.0
	for i in range(0,m):
		if a[i][0] == 1:
			numberOfPos = numberOfPos+1
			seen[a[i][index]] = seen[a[i][index]] + 1		
	for i in range(0,m):
		a[i][index] = seen[a[i][index]]/numberOfPos
	return a
		
'''	
A= makeArray()
A = np.array(A)

le = LabelEncoder()
for i in range(0,12):
	le.fit(A[:,i])
	A[:,i] = le.transform(A[:,i])
	print "i: "+str(i)
	print len(le.classes_)
	
A.astype(int)
print A.shape
saveData(A,"300Data")
'''

'''
b = loadArray("300Data")

enc = OneHotEncoder()
print "fitting"

x = b[:,1:b.shape[0]]
y = b[:,0]

enc.fit(x)
x = enc.transform(x)
ohe = open("onehotencoding300.npz",'wb')
save_sparse_csr(ohe,x)
train_y = open("train_y300.npz",'wb')
np.save(train_y,y)
'''

numberOfClasses = [0,24,555583,35,370,3,83928,11,6,4,3,19]
b = loadArray("300Data")
b = b.astype(float)
m = b.shape[0]
n = b.shape[1]
print m
print n
for i in range(1,12):
	b = makeFeatures(b,i,numberOfClasses[i],m)
print b[0]
x = b[:,1:b.shape[0]]
y = b[:,0]
np.save("train_Perc",x)




