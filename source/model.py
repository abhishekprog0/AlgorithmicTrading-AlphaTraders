# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-04-19 20:45:24
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-04-29 16:31:18
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import pandas as pd
import numpy as np

from model_params import input_size


class Network(nn.Module):
	def __init__(self,input_size):
		super(Network,self).__init__()

		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

		self.layer1 = nn.Linear(input_size,100) 
		self.layer2 = nn.Linear(100,150)
		self.layer3 = nn.Linear(150,100)
		self.layer4 = nn.Linear(100,80)
		self.layer5 = nn.Linear(80,60)
		self.layer6 = nn.Linear(60,40)
		self.layer7 = nn.Linear(40,20)
		self.layer8 = nn.Linear(20,1)

	def forward(self, data):
		out = self.relu(self.layer1(data))
		out = self.relu(self.layer2(out))
		out = self.relu(self.layer3(out))
		out = self.relu(self.layer4(out))
		out = self.relu(self.layer5(out))
		out = self.relu(self.layer6(out))
		out = self.relu(self.layer7(out))
		out = self.layer8(out)
		return out

'''Testing the network'''
if __name__ == '__main__':
	x = torch.randn((5))
	test = Network(input_size)
	print (test.forward(x).shape)