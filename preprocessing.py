import sys
import codecs
import csv

def extractTime(date):
	return date[8:10]

def formatFloorPrice(price):
	try:
		price = int(price)
	except:
		return 0
	if price == 0:
		return 0
	elif price<50:
		return 1
	elif price<100:
		return 2
	elif price<200:
		return 3
	else:
		return 4
		
def process(line):
	line = line.split("\t")
	delete = [29,28,27,26,25,23,17,16,15,10,9,8,7,6,5,4,3,1]
	for i in delete:
		del line[i]
	return line


f = open('x_train.csv','ab')
wr = csv.writer(f)
trainFile = r"./train.log.txt"
		
with open(trainFile) as infile:
	for line in infile:
		line = line.decode('utf-8', 'ignore')
		try:
			line = process(line)
		except:
			print line
		wr.writerow(line)
		if not line:
			break
				
f.close()