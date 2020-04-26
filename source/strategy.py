# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-04-25 13:57:44
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-04-25 22:34:44


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


from execution_source.Execution import vwapExecute

class ExecuteTrade(object):
	"""docstring for executeTrade"""
	def __init__(self):
		super(ExecuteTrade, self).__init__()
		self.ticker = ['MA', 'AMZN', 'BAC', 'MRK', 'WFC', 'INTC', 'T', 'GOOG', 'HD', 'CSCO', 'PFE', 'VZ', 'UNH', 'NKE', 'NFLX' ,'JNJ', 'MCD', 'AAPL', 'JPM', 'MSFT'] 
		self.trade = {}

	def executeTrade(self,data,close_prices):
		prev_weights = data['old'].detach().numpy().reshape(-1)
		new_weights = data['new'].detach().numpy().reshape(-1)
		wealth = data['wealth']
		notional = (new_weights - prev_weights) * wealth 
		num_trade = notional / close_prices
		num_trade = num_trade.astype(int)
		
		i = 0
		for name in self.ticker:
			side = ''
			if num_trade[i] > 0:
				side = "b"
			else:
				side = "s"
			self.trade[name] = (side, num_trade[i])
			i+=1

	def vwap(self):
		i = 'MSFT'
		trade_file = i+'_trade.zip'
		quote_file = i+'_quote.zip'

		print(self.trade)
		order = self.trade[i][1]
		side  = self.trade[i][0]
		print('\n************************************')
		print(i)
		print(order)
		print('************************************\n')
		
		vwapExecute(trade_file,quote_file,order,side)


#unit test
if __name__ == '__main__':
	getData('x')
