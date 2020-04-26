import numpy as np
import torch
import sys
import math
sys.path.insert(0, '../')

from model_params import transaction_commission

class Loser():
	def __init__(self,output_size):
		self.dim = output_size
		self.weight_buffer = list(torch.zeros(1,20))
		self.reset()

	def reset(self):
		self.wealth = 1e6
		self.wealth_history = []
		self.return_history = []		

	def resetBuffer(self):
		self.weight_buffer = list(torch.zeros(1,20))

	def updateSummary(self,loss):
		reward = -1 * loss
		self.wealth = self.wealth * math.exp(reward)
		self.wealth_history.append(self.wealth)
		self.return_history.append(math.exp(reward))

	def predict(self,data):
		close = data[-1][-1][-1]
		weights = torch.zeros(self.dim)
		weights[torch.argmin(close)] = 1
		weights = weights.view(1,weights.shape[0])
		self.weight_buffer.append(weights)
		return weights
	
	def loss_function(self,w1,w2,y):
		transaction_cost = 1 - torch.sum(torch.abs(w2-w1))*transaction_commission
		portfolio_return = torch.sum(w2*y)
		loss = -1 * torch.log(portfolio_return*transaction_cost)
		loss = torch.mean(loss)
		return loss


'''Unit testing the file'''

if __name__ == '__main__':
	test = Loser(20)
	data = torch.randn(1,1,20,6)
	out = test.predict(data)
	p_weights = test.weight_buffer[-1]
	loss = test.loss_function(p_weights,out,torch.randn(20))
	test.updateSummary(loss)
	print (test.weight_buffer)

