import sys
import codecs
import csv

def extractTime(date):
	return date[8:9]

def formatFloorPrice(price):
	price = int(price)
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
		
def process(line,bidID):
	line = line.split("\t")
	if line[0] in bidID:
		line[0] = "1"
	else:
		line[0] = "0"
	line[16] = formatFloorPrice(line[16])
	delete = [20,18,17,10,9,3,2]
	for i in delete:
		del line[i]
	line[1] = extractTime(line[1])
	return line

bidID = ["a"]
with open("clk.20130612.txt", encoding='utf', errors='ignore') as infile:
	for line in infile:
		line = line.split("\t")
		bidID.append(line[0])
		
f = open('trainingDataLR.txt','w')
wr = csv.writer(f, quoting=csv.QUOTE_ALL)
with open("bid.20130612.txt", encoding='utf-8', errors='ignore') as infile:
	i = 0
	for line in infile:
		i = i+1
		line = process(line,bidID)
		wr.writerow(line)
		if not line:
			break