import torch
import torchvision
import torch.optim as optim 
import torch.nn as nn
import pandas as pd
import numpy as np

from model_params import input_channels


class PolicyNetwork(nn.Module):
	def __init__(self,input_channels):
		super(PolicyNetwork,self).__init__()
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()
		self.conv1 = nn.Conv2d(in_channels=1,out_channels=2,kernel_size=(1,2),stride=1,padding=0)
		self.conv2 = nn.Conv2d(in_channels=2,out_channels=48,kernel_size=(1,5),stride=1,padding=0)
		self.conv3 = nn.Conv2d(in_channels=49,out_channels=1,kernel_size=(1,1),stride=1,padding=0)
		self.layer1 = nn.Linear(1*20*1,20)

		#Investment only on cash not in any equity
		self.weight_buffer = list(torch.zeros(1,20))

	def resetBuffer(self):
		self.weight_buffer = list(torch.zeros(1,20))

	def num_flatten_features(self,x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

	def forward(self,data,w_previous):
		out = self.relu(self.conv1(data))
		out = self.relu(self.conv2(out))
		#Adding the previous weight vector for better learning
		w_previous = w_previous.view(-1,1,20,1)
		out = torch.cat((out,w_previous),1)
		out = self.relu(self.conv3(out))
		#Flattening features to feed in linear layer
		out = out.view(-1,self.num_flatten_features(out))
		out = self.softmax(self.layer1(out))
		self.weight_buffer.append(out)
		return out

if __name__ == '__main__':
	test = PolicyNetwork(input_channels)
	x = torch.randn((1,1,20,6))
	# test.weight_buffer.append(torch.randn((1,14)))
	print (test.forward(x,test.weight_buffer[-1]).shape)
	# print test.weight_buffer
	# test.resetBuffer()
	# print test.weight_buffer