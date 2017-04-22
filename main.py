import random
import numpy as np

class LinPlayer():
	def __init__(self,v,B,T,probX):
		self.values = v
		self.maxBid = 3*B/T
		self.state = [T,B]
		self.avgCTR = 0.0
		for i in range(0,3):
			self.avgCTR = self.avgCTR + probX[i]*v[i]
		self.basicBid = 2.0
	
	def resetState(self,T,B):
		self.state[0] = T
		self.state[1] = B
		
	def updateState(self,action):
		self.state[0] = self.state[0]-1
		self.state[1] = self.state[1]-action
		
	def getCTR(self,featureVec):
		ctr = 0.0
		for i in range(0,3):
			ctr = ctr + self.values[i]*featureVec[i]/10
		return ctr
		
	def getAction(self,featureVec):
		if self.state[0] <= 0:
			return 0.0
		else:
			action = min(self.state[1],self.basicBid*self.getCTR(featureVec)/self.avgCTR)
			return action
			
class QPlayerFeatures():

	def __init__(self,v,T,B,mB):
		self.values = v
		self.maxBid = mB
		self.state = [T,B]
		self.qFunction = [[[[0.0 for b in range(self.maxBid)]for x in range(3)]for y in range(B+1)]for z in range(T+1)]
		
	def resetState(self,T,B):
		self.state[0] = T
		self.state[1] = B
		
	def getCTR(self,featureVec):
		sum = 0
		for i in range(0,3):
			sum = self.values[i]*featureVec[i]
		return sum
		
	def getAction(self,featureVec,reservePrice,epsilon):
		if featureVec[0] == 1:
			f = 0
		elif featureVec[1] == 1:
			f = 1
		else:
			f = 2
		if random.random() < epsilon:
			action = random.randrange(reservePrice,min(self.state[1],maxBid))
		elif reservePrice > self.state[1]:
			action = 0
		else:
			action = 0
			expectedReturn = 0
			for i in range(reservePrice,min(self.state[1],maxBid)):
				if self.qFunction[self.state[0]][self.state[1]][f][i] > expectedReturn:
					expectedReturn = self.qFunction[self.state[0]][self.state[1]][f][i]
					action = i
		return action
	
	def updateState(self,action):
		self.state = [self.state[0]-1,self.state[1]-action]
		
	def updateQFunction(self,previousState,featureVec,action,nextState,nextFeatureVec,reward,learningRate,reservePrice,gamma):
		if featureVec[0] == 1:
			f = 0
		elif featureVec[1] == 1:
			f = 1
		else:
			f = 2
		if nextFeatureVec[0] == 1:
			f2 = 0
		elif nextFeatureVec[1] == 1:
			f2 = 1
		else:
			f2 = 2
		a2 = self.getAction(nextFeatureVec,reservePrice,0.0)
		self.qFunction[previousState[0]][previousState[1]][f][action] = reward + learningRate*(gamma*self.qFunction[nextState[0]][nextState[1]][f2][a2] - self.qFunction[previousState[0]][previousState[1]][f][action])
		
		
class QPlayerCTR():

	def __init__(self,v,T,B,mB):
		self.values = v
		self.maxBid = mB
		self.state = [T,B]
		self.qFunction = [[[[0.0 for b in range(maxBid)]for x in range(4)] for y in range(B)] for z in range(T)]
		
	def resetState(self,T,B):
		self.state[0] = T
		self.state[1] = B
		
	def getBucket(self,CTR):
		if CTR<0.05:
			return 0
		elif CTR<0.1:
			return 1
		elif CTR<0.15:
			return 2
		else:
			return 3
		
	def getCTR(self,featureVec):
		sum = 0
		for i in range(0,3):
			sum = self.values[i]*featureVec[i]
		return sum
		
	def getAction(self,featureVec,reservePrice,epsilon):
		f = self.getBucket(self.getCTR(featureVec))
		if random.random() < epsilon:
			action = random.randrange(reservePrice,min(self.state[1],maxBid))
		elif reservePrice > self.state[1]:
			action = 0
		else:
			action = 0
			expectedReturn = 0
			for i in range(0,min(self.state[1],maxBid)):
				if self.qFunction[self.state[0]][self.state[1]][f][i] > expectedReturn:
					expectedReturn = self.qFunction[self.state[0]][self.state[1]][f][i]
					action = i
		return action
	
	def updateState(self,action):
		self.state = [self.state[0]-1,self.state[1]-action]
		
	def updateQFunction(self,previousState,featureVec,action,nextState,nextFeatureVec,reward,learningRate,reservePrice,gamma):
		f = self.getBucket(self.getCTR(featureVec))
		f2 = self.getBucket(self.getCTR(nextFeatureVec))
		a2 = self.getAction(nextFeatureVec,reservePrice,0.0)
		self.qFunction[previousState[0]][previousState[1]][f][action] = reward + learningRate*(gamma*self.qFunction[nextState[0]][nextState[1]][f2][a2] - self.qFunction[previousState[0]][previousState[1]][f][action])
		
class QPlayerCTRNoise():

	def __init__(self,v,T,B,P):
		self.values = v
		self.maxBid = 3*B/T
		self.state = [T,B]
		self.qFunction = [[[[0.0 for b in range(maxBid)]for x in range(4)] for y in range(B)] for z in range(T)]
		
	def resetState(self,T,B):
		self.state[0] = T
		self.state[1] = B
		
	def getBucket(self,CTR):
		if CTR<0.05:
			return 0
		elif CTR<0.1:
			return 1
		elif CTR<0.15:
			return 2
		else:
			return 3
		
	def getCTRNoise(self,featureVec):
		sum = 0
		for i in range(0,3):
			sum = self.values[i]*featureVec[i]
		noise = np.random.normal(0,1,100)
		return sum+noise
		
	def getAction(self,featureVec,reservePrice,epsilon):
		f = self.getBucket(self.getCTR(featureVec))
		if random.random() < epsilon:
			action = random.randrange(reservePrice,min(self.state[1],maxBid))
		elif reservePrice > self.state[1]:
			action = 0
		else:
			action = 0
			expectedReturn = 0
			for i in range(0,min(self.state[1],maxBid)):
				if self.qFunction[self.state[0]][self.state[1]][f][i] > expectedReturn:
					expectedReturn = self.qFunction[self.state[0]][self.state[1]][f][i]
					action = i
		return i
	
	def updateState(self,action):
		self.state = [self.state[0]-1,self.state[1]-action]
		
	def updateQFunction(self,previousState,featureVec,action,nextState,nextFeatureVec,reward,learningRate,reservePrice,gamma):
		f = self.getBucket(self.getCTR(featureVec))
		f2 = self.getBucket(self.getCTR(nextFeatureVec))
		a2 = self.getAction(nextFeatureVec,reservePrice,0.0)
		self.qFunction[previousState[0]][previousState[1]][f][action] = reward + learningRate*(gamma*self.qFunction[nextState[0]][nextState[1]][f2][a2] - self.qFunction[previousState[0]][previousState[1]][f][action])
		
class FriendQPlayerCTR():

	def __init__(self,v,T,B,P):
		self.values = v
		self.maxBid = 3*B/T
		self.state = [T,B]
		self.qFunction = [[[[[[0.0 for b1 in range(maxBid)] for b2 in range(maxBid)] for b3 in range(maxBid)]for x in range(4)] for y in range(B)] for z in range(T)]
		
	def resetState(self,T,B):
		self.state[0] = T
		self.state[1] = B
		
	def getBucket(self,CTR):
		if CTR<0.05:
			return 0
		elif CTR<0.1:
			return 1
		elif CTR<0.15:
			return 2
		else:
			return 3
		
	def getCTRNoise(self,featureVec):
		sum = 0
		for i in range(0,3):
			sum = self.values[i]*featureVec[i]
		return sum
		
	def getAction(self,featureVec,reservePrice,epsilon):
		f = self.getBucket(self.getCTR(featureVec))
		if random.random() < epsilon:
			action = random.randrange(reservePrice,min(self.state[1],maxBid))
		elif reservePrice > self.state[1]:
			action = 0
		else:
			action = 0
			expectedReturn = 0
			for i in range(0,min(self.state[1],maxBid)):
				for j in range(0,min(self.state[1],maxBid)):
					for k in range(0,min(self.state[1],maxBid)):
						if self.qFunction[self.state[0]][self.state[1]][f][i][j][k] > expectedReturn:
							expectedReturn = self.qFunction[self.state[0]][self.state[1]][f][i]
							action = [i,j,k]
		return action
	
	def updateState(self,action):
		self.state = [self.state[0]-1,self.state[1]-action[0]]
		
	def updateQFunction(self,previousState,featureVec,action,nextState,nextFeatureVec,reward,learningRate,reservePrice,gamma):
		[i,j,k] = action
		f = self.getBucket(self.getCTR(featureVec))
		f2 = self.getBucket(self.getCTR(nextFeatureVec))
		a2 = self.getAction(nextFeatureVec,reservePrice,0.0)
		[i2,j2,k2] = a2
		self.qFunction[previousState[0]][previousState[1]][f][i][j][k] = reward + learningRate*(gamma*self.qFunction[nextState[0]][nextState[1]][f2][i2][j2][k2] - self.qFunction[previousState[0]][previousState[1]][f][i][j][k])
		
		
class AuctionUni:

	def __init__(self,P1,P2,P3,T,probX,learning,epsilon):
		featureVec = self.generateNextBid(probX)
		self.cumulativeReward = 0
		reservePrice = 0
		while(T>0):
			T = T-1
			featureVec = self.generateNextBid(probX)
			a1 = P1.getAction(featureVec,reservePrice,epsilon)
			a2 = P2.getAction(featureVec)
			a3 = P3.getAction(featureVec)
			winner,payment = self.determineWinnerAndPayment(a1,a2,a3)
			nextFeatureVec = self.generateNextBid(probX)
			r1 = 0
			r2 = 0
			r3 = 0
			if winner == 0:
				r1 = self.generateReward(P1,featureVec)
				self.cumulativeReward = self.cumulativeReward + 1
				P2.updateState(0)
				P3.updateState(0)
			elif winner == 1:
				r2 = self.generateReward(P2,featureVec)
				P2.updateState(payment)
				P3.updateState(0)
			else:
				r3 = self.generateReward(P3,featureVec)
				P2.updateState(0)
				P3.updateState(payment)
			if learning:
				previousState = P1.state
				if winner == 0:
					P1.updateState(payment)
				else:
					P1.updateState(0)
				nextState = P1.state
				P1.updateQFunction(previousState,featureVec,a1,nextState,nextFeatureVec,r1,0.05,1,0.95)
				
	def resetState(self,T,B):
		self.T = T
		self.B = B
		
	def generateReward(self,player,feature):
		CTR = player.getCTR(feature)
		if random.random<CTR:
			return 1
		else:
			return 0
			
	def generateNextBid(self,probX):
		a = random.random()
		if a < probX[0]:
			return [1,0,0]
		elif a < probX[1]:
			return [0,1,0]
		else:
			return [0,0,1]
		
	def determineWinnerAndPayment(self,a1,a2,a3):
		winner = 0
		payment = 0
		if a1 > a2:
			winner = 0
			payment = a1
		else:
			winner = 1
			payment = a2
		if payment > a3:
			if a3>a1:
				payment = a3
		else:
			winner = 2
		return winner,payment
		
class AuctionMulti:

	def __init__(self,P1,P2,P3,T,probX,learning,epsilon):
		featureVec = self.generateNextBid(probX)
		self.cumulativeReward = 0
		reservePrice = 0
		while(T>0):
			T = T-1
			featureVec = self.generateNextBid(probX)
			a1 = P1.getAction(featureVec,reservePrice,epsilon)
			a2 = P2.getAction(featureVec,reservePrice,epsilon)
			a3 = P3.getAction(featureVec,reservePrice,epsilon)
			winner,payment = self.determineWinnerAndPayment(a1[0],a2[0],a3[0])
			nextFeatureVec = self.generateNextBid(probX)
			r1 = 0
			r2 = 0
			r3 = 0
			if winner == 0:
				r1 = self.generateReward(p1,featureVec)
				self.cumulativeReward = self.cumulativeReward + 1
			elif winner == 1:
				r2 = self.generateReward(p2,featureVec)
			else:
				r3 = self.generateReward(p3,featureVec)
			if learning:
				previousState = P1.state
				if winner == 0:
					P1.updateState(payment)
				else:
					P1.updateState(0.0)
				nextState = P1.state
				P1.updateQFunction(previousState,featureVec,a1,nextState,nextFeatureVec,r1,0.05,1,0.95)
				previousState = P2.state
				if winner == 1:
					P2.updateState(payment)
				else:
					P2.updateState(0.0)
				nextState = P3.state
				P3.updateQFunction(previousState,featureVec,a1,nextState,nextFeatureVec,r1,0.05,1,0.95)
				previousState = P3.state
				if winner == 2:
					P3.updateState(payment)
				else:
					P3.updateState(0.0)
				nextState = P3.state
				P3.updateQFunction(previousState,featureVec,a1,nextState,nextFeatureVec,r1,0.05,1,0.95)
		
	
	def generateReward(self,player,feature):
		CTR = player.getCTR(features)
		if random.random<CTR:
			return 1
		else:
			return 0
			
	def generateNextBid(self,probX):
		a = random.random()
		if a < probX[0]:
			return [1,0,0]
		elif a < probX[1]:
			return [0,1,0]
		else:
			return [0,0,1]
		
	def determineWinnerAndPayment(self,a1,a2,a3):
		winner = 0
		payment = 0
		if a1 > a2:
			winner = 0
			payment = a1
		else:
			winner = 1
			payment = a2
		if payment > a3:
			if a3>a1:
				payment = a3
		else:
			winner = 2
		return winner,payment	
		
numberOfEpisodes = 200000
T = 100
c = 0.5
B = int(3*100*c)
maxBid = 5
probX = [0.33,0.66,1]
v1 = [0.15,0.0,0.0]
v2 = [0.15,0.0,0.0]
v3 = [0.0,0.1,0.1]
player1 = QPlayerFeatures(v1,T,B,20)
player2 = LinPlayer(v2,T,B,[0.33,0.33,0.33])
player3 = LinPlayer(v3,T,B,[0.33,0.33,0.33])
		
for i in range(numberOfEpisodes):
	a = AuctionUni(player1,player2,player3,T,probX,True,0.1)
	player1.resetState(T,B)
	player2.resetState(T,B)
	player3.resetState(T,B)

clicks = 0	
for j in range(100):
	a = AuctionUni(player1,player2,player3,T,probX,False,0.1)
	player1.resetState(T,B)
	player2.resetState(T,B)
	player3.resetState(T,B)
	clicks = clicks + a.cumulativeReward	

	
print clicks