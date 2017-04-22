import matplotlib.pyplot as plt
import numpy as np
a = [[192257,7674],[63,6]]
b = [[192320,7680],[0,0]]
c = [[192320,7680],[0,0]]

def plotConfusionMatrix(a,i):
	a = np.array(a)
	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(np.array(a), cmap=plt.cm.jet, 
                	interpolation='nearest')

	width, height = a.shape

	for x in xrange(width):
		for y in xrange(height):
			ax.annotate(str(a[x][y]), xy=(y, x), 
						horizontalalignment='center',
						verticalalignment='center')

	cb = fig.colorbar(res)
	alphabet = 'AB'
	plt.xticks(range(width), alphabet[:width])
	plt.yticks(range(height), alphabet[:height])
	plt.savefig('confusion_matrix'+str(i)+'.png', format='png')


plotConfusionMatrix(a,1)
plotConfusionMatrix(b,2)
plotConfusionMatrix(c,3)