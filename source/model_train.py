import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import visdom
import argparse
import pandas as pd
import numpy as np

from model_params import input_size
from model_params import sequence_size
from model_params import train_size
from model_params import learning_rate
from model_params import num_epochs
from model_params import batch_size

from model import Network
from utils import Plotter
from utils import DataProcessing

class Train(nn.Module):
	def __init__(self):
		super(Train,self).__init__()
		self.input_size = input_size
		self.batch_size = batch_size
		self.sequence_size = sequence_size
		self.learning_rate = learning_rate
		'''Constructing the network'''
		self.network = Network(self.input_size)
		self.loss_function = nn.MSELoss(reduction='mean')
		'''Defining the optimizer'''
		self.parameters = self.network.parameters()
		self.optimizer = optim.Adam(self.parameters, lr = self.learning_rate)


	'''Calculating the loss for the network'''
	def loss(self,predicted,target):
		return self.loss_function(predicted,target)

'''getting batches and flattening the input tensor'''
def getBatch(net,data,target,batch):
	start_pos = batch * net.batch_size
	end_pos = start_pos + net.batch_size
	x = torch.from_numpy(data[start_pos:end_pos]).float()
	y = torch.from_numpy(target[start_pos:end_pos]).float()
	# x = x.view(x.shape[0],x.shape[1]*x.shape[2])
	y = y.view(y.shape[0],1)
	return x,y

def trainNetwork(net,data,target,epoch):
	num_batches = (len(data)//net.batch_size)
	total_loss = 0
	for batch in range(num_batches):

		X,Y = getBatch(net,data,target,batch)
		# print Y.shape
		out = net.network.forward(X)
		loss = net.loss_function(out,Y)
		net.optimizer.zero_grad()
		loss.backward()
		net.optimizer.step()

		total_loss +=loss.item()
	
		if epoch == 74:
			print out*10

	print('Epoch: {} LOSS: {}'.format(epoch, total_loss/num_batches))
	plotter.plot('Loss', 'Train', 'SBI2 Training Loss', epoch, total_loss/num_batches)

	return net
	


if __name__ == '__main__':
	global plotter
	plotter = Plotter(env_name='Thesis Project')

	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", type=str, default="../data", help="path to the dataset")
	parser.add_argument("--file", type=str, default="combined2.csv")
	args = parser.parse_args()
	file_name = args.datapath + '/' + args.file

	data = DataProcessing(file_name,train_size)
	train_data,train_target = data.trainingData()


	net = Train()

	for epoch in range(num_epochs):
		net = trainNetwork(net,train_data,train_target,epoch)

	torch.save(net.network.state_dict(),'../saved_models/model_JINDAL.pt')
	