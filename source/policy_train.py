import torch
import torchvision
import visdom
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import statistics

from bench_mark.loser import Loser
from bench_mark.winner import Winner
from bench_mark.ucrp import UCRP

from policy_utils import DataProcessing
from policy_utils import Plotter
from policy_utils import Multiple_Plotter
from policy_network import PolicyNetwork


from model_params import window_size
from model_params import epochs
from model_params import input_channels
from model_params import policy_learning_rate
from model_params import train_size
from model_params import policy_batch_size as batch_size
from model_params import transaction_commission
from model_params import bench_mark_output_size

class Train(nn.Module):
	def __init__(self,learning_rate,input_channels):
		super(Train,self).__init__()

		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.epochs = epochs
		self.reset()
		self.network = PolicyNetwork(input_channels)
		self.parameters = self.network.parameters()
		self.optimizer = optim.Adam(self.parameters,lr=self.learning_rate)


	def reset(self):
		self.wealth = 10e4
		self.loss = 0
		self.total_reward = 0
		self.wealth_history = []
		self.return_history = []

	def updateSummary(self,loss):
		self.loss += loss
		reward = -1 * loss
		self.wealth = self.wealth * math.exp(reward)
		self.total_reward += reward
		self.wealth_history.append(self.wealth)
		self.return_history.append(math.exp(reward))

	def getBatch(self,data,target,batch):
		start_pos = batch * self.batch_size
		end_pos = start_pos + self.batch_size
		x = torch.from_numpy(data[start_pos:end_pos]).float()
		y = torch.from_numpy(target[start_pos:end_pos]).float()
		# x = x.view(x.shape[0],x.shape[1]*x.shape[2])
		# y = y.view(y.shape[0],1)
		
		#Normalizing X by the last real day price
		x = x / x[-1][-1][-2]
		return x,y

	def loss_function(self,w1,w2,y):
		transaction_cost = 1 - torch.sum(torch.abs(w2-w1))*transaction_commission
		portfolio_return = torch.sum(w2*y)
		loss = -1 * torch.log(portfolio_return*transaction_cost)
		loss = torch.mean(loss)
		return loss

def trainNetwork(net,train_data,train_target,iterations):
	num_batches = (len(train_data)//net.batch_size)
	for batch in range(num_batches):
		iterations+=1

		x,y = net.getBatch(train_data,train_target,batch)
		
		#Forward Propagation
		previous_weights = net.network.weight_buffer[-1]
		new_weights = net.network.forward(x,previous_weights)

		#Calculating Loss
		loss = net.loss_function(previous_weights,new_weights,y)
		
		#Back Propagation
		net.optimizer.zero_grad()
		loss.backward(retain_graph=True)
		net.optimizer.step()

		# print net.wealth*torch.exp(-1*loss.item())
		net.updateSummary(loss.item())
		plotter.plot('Wealth', 'iterations', 'Policy Wealth', iterations, net.wealth)
		print('Iteration:{} Wealth:{}').format(iterations, net.wealth)
	print('Wealth:{} Loss:{}').format(net.wealth,net.loss/num_batches)
	return net


def backTest(net,winner,loser,ucrp,test_data,test_target,iterations):
	num_batches = (len(test_data)//net.batch_size)
	for batch in range(num_batches):
		iterations += 1

		x,y = net.getBatch(test_data,test_target,batch)

		#Policy Forward Propagation
		previous_weights = net.network.weight_buffer[-1]
		new_weights = net.network.forward(x,previous_weights)
		policy_loss = net.loss_function(previous_weights,new_weights,y)
		net.updateSummary(policy_loss.item())

		#Follow the winner
		winner_previous_weights = winner.weight_buffer[-1]
		new_weights_winner = winner.predict(x)
		winner_loss = winner.loss_function(winner_previous_weights, new_weights_winner, y)
		winner.updateSummary(winner_loss)

		#Follow the loser
		loser_previous_weights = loser.weight_buffer[-1]
		new_weights_loser = loser.predict(x)
		loser_loss = loser.loss_function(loser_previous_weights, new_weights_loser, y)
		loser.updateSummary(loser_loss)

		#Follow UCRP
		ucrp_previous_weights = ucrp.weight_buffer[-1]
		new_weights_ucrp = ucrp.predict()
		ucrp_loss = ucrp.loss_function(ucrp_previous_weights, new_weights_ucrp, y)
		ucrp.updateSummary(ucrp_loss,iterations)


		plot_x = [net.wealth, winner.wealth, loser.wealth,ucrp.wealth]
		backtest_plotter.plot(['Policy_Wealth','Winner_wealth','Loser Wealth','UCRP Wealth'], 'Days', 'Policy Wealth', iterations, plot_x)	

def calculateSharpeRation(net,winner,loser,ucrp):
	sharpe_ratio_net = ((statistics.mean(net.return_history)-1)/100)/statistics.stdev(net.return_history)	
	sharpe_ratio_winner = ((statistics.mean(winner.return_history)-1)/100)/statistics.stdev(winner.return_history)	
	sharpe_ratio_loser = ((statistics.mean(loser.return_history)-1)/100)/statistics.stdev(loser.return_history)	
	sharpe_ratio_ucrp = ((statistics.mean(ucrp.return_history)-1)/100)/statistics.stdev(ucrp.return_history)	

	print sharpe_ratio_net
	print sharpe_ratio_winner
	print sharpe_ratio_loser
	print sharpe_ratio_ucrp


if __name__ == '__main__':
	global plotter
	plotter = Plotter(env_name='Policy Network')

	global backtest_plotter
	backtest_plotter = Multiple_Plotter(env_name='Test')

	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str, default='train', help='Please select mode training or test')
	parser.add_argument("--datapath", type=str, default="../data", help="path to the dataset")
	parser.add_argument("--file", type=str, default="combined2.csv", help='Please give path to the data files')
	parser.add_argument("--models", type=str, default="../saved_models", help="path to the dataset")
	args = parser.parse_args()
	file_name = args.datapath + '/' + args.file
	mode = args.mode
	
	#Data utility object
	# data = DataProcessing(file_name,train_size,args.models)
	data = DataProcessing(file_name,0.8,args.models)

	iterations = 0

	#Training model
	if mode == 'train':
		net = Train(policy_learning_rate,input_channels)
		train_data, train_target = data.trainingData()

		for epoch in range(net.epochs):
			print '--------------------------'
			print('Epoch: {}').format(epoch+1)
			print '-------------------------'
			net = trainNetwork(net,train_data,train_target,iterations)
			net.reset()
			net.network.resetBuffer()	

		torch.save(net.network.state_dict(),'../saved_models/policy_network.pt')

	#Testing the model
	elif mode == 'test':
		try:
			print"\n======================================="
			print "             Loading model              "
			print "========================================\n"
			net = Train(policy_learning_rate,input_channels)
			net.network.load_state_dict(torch.load('../saved_models/policy_network.pt'))
			loser = Loser(bench_mark_output_size)
			winner = Winner(bench_mark_output_size)
			ucrp = UCRP(bench_mark_output_size)
		except Exception as e:
			print e
			print 'Please give the correct path to the saved model'
			
		test_data, test_target = data.testingData()

		backTest(net,winner,loser,ucrp,test_data,test_target,iterations)

		calculateSharpeRation(net,winner,loser,ucrp)
		
