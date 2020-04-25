import torch
import numpy as np
import pandas as pd
from visdom import Visdom

from model_params import window_size
from model import Network
from model_params import input_size

class Multiple_Plotter(object):
	def __init__(self,env_name='main'):
		self.viz = Visdom()
		self.env = env_name
		self.plots = {}

	def plot(self,var_name,split_name,title_name,x,y):
		name = 'Loss Plot'
		# print np.array([x]),np.array(y)
		if name not in self.plots:
			self.plots[name] = self.viz.line(X=np.array([x]),Y=np.array([y[0]]),env=self.env, opts=dict(
			legend=var_name,title=title_name,xlabel='Iterations',ylabel='Wealth'),name='Policy')
		else:
			self.viz.line(X=np.array([x]), Y=np.array([y[0]]), env=self.env, win=self.plots[name],name='Policy_Wealth', update = 'append')
			self.viz.line(X=np.array([x]), Y=np.array([y[1]]), env=self.env, win=self.plots[name],name='Winner', update = 'append')
			self.viz.line(X=np.array([x]), Y=np.array([y[2]]), env=self.env, win=self.plots[name],name='Loser', update = 'append')
			self.viz.line(X=np.array([x]), Y=np.array([y[3]]), env=self.env, win=self.plots[name],name='UCRP', update = 'append')


class Plotter(object):
	"""Plots to Visdom"""
	def __init__(self, env_name='main'):
		self.viz = Visdom()
		self.env = env_name
		self.plots = {}
	def plot(self, var_name, split_name, title_name, x, y):
		if var_name not in self.plots:
			self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
				legend=[split_name],
				title=title_name,
				xlabel='Iterations',
				ylabel=var_name
			))
		else:
			self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


class DataProcessing(object):
	def __init__(self,file_name,train_size,path):
		self.df = pd.read_csv(file_name)
		self.df = self.df.drop(columns=['Date'])
		self.train_size = train_size
		self.index = int(len(self.df) * self.train_size)
		self.window = window_size
		self.train_data = self.df.iloc[:self.index]
		self.test_data = self.df.iloc[self.index:]

		self.trained_models = self.getModels(path)

	def getModels(self,path):
		name = ['MA','AMZN','BAC','MRK','WFC','INTC','T','GOOG','HD','CSCO','PFE','VZ','UNH','NKE','NFLX','JNJ','BRK-B','MCD','AAPL','JPM','MSFT']

		models = []
		print ("---------------------------------------------")
		print ("Loading Saved Models")
		print ("---------------------------------------------\n")
		
		for n in name:
			model_name = str(path)+'/model_'+n+'.pt'
			temp = Network(input_size)
			print (model_name)
			temp.load_state_dict(torch.load(model_name))
			temp.eval()
			models.append(temp)		
		return models

	def trainingData(self):
		print ("\n---------------------------------------------")
		print ("Loading Training Data")
		print ("---------------------------------------------\n")
		attributes = []
		target = []
		#Creating Window
		for i in range( (len(self.train_data)//self.window)*self.window - self.window - 1 ):
			#X is the price tensor that contains the history of the market
			#Y is the relative price vector i.e vt/vt-1

			x = np.array(self.train_data.iloc[i:i+self.window],np.float64)
			y = np.array(self.train_data.iloc[i+self.window],np.float64)
			y = y/x[-1]
			
			#reshaping the price tensor
			x = x.T
			
			predicted = [] 

			#Predicting next and concatenating
			normalization_factor = 10
			for	j in range(len(x)):
				input_data = torch.from_numpy(x[j]/normalization_factor).float()
				pred = self.trained_models[j].forward(input_data) * normalization_factor
				predicted.append(pred.item())
			
			x = np.concatenate((x,np.array(predicted).reshape(21,1)),axis=1).reshape(1,21,6)

			attributes.append(x)
			target.append(y)
		
		return np.array(attributes),np.array(target)


	def testingData(self):
		print ("\n---------------------------------------------")
		print ("Loading Testing Data")
		print ("---------------------------------------------\n")
		attributes = []
		target = []

		#Creating Window
		for i in range( (len(self.test_data)//self.window)*self.window - self.window - 1 ):
			#X is the price tensor that contains the history of the market
			#Y is the relative price vector i.e vt/vt-1
			x = np.array(self.test_data.iloc[i:i+self.window],np.float64)
			y = np.array(self.test_data.iloc[i+self.window],np.float64)
			y = y/x[-1]
			
			#reshaping the price tensor
			x = x.T
			
			predicted = [] 
			#Predicting next and concatenating
			for	j in range(len(x)):
				normalization_factor = 10
				# if j in[0,1,2,3,6,8,9,10,12,13]:
				# 	normalization_factor = 10
				# elif j in [4,5,11]:
				# 	normalization_factor = 50
				# else:
				# 	normalization_factor = 250

				input_data = torch.from_numpy(x[j]/normalization_factor).float()
				pred = self.trained_models[j].forward(input_data) * normalization_factor
				predicted.append(pred.item())
			
			x = np.concatenate((x,np.array(predicted).reshape(14,1)),axis=1).reshape(1,14,6)

			attributes.append(x)
			target.append(y)
		
		return np.array(attributes),np.array(target)


'''Testing utils for policy network'''
if __name__ == '__main__':
	x = DataProcessing('../data/combined2.csv',0.8,'../saved_models')
	attributes,target = x.testingData()
	
	print (attributes[-1])
	# print attributes[0]
	# print target[0:1].shape
	# print attributes[430:440]