# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-04-10 20:41:14
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-04-10 21:25:20

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime

class DataReader(object):
	"""docstring for DataReader"""
	def __init__(self, ticker):
		super(DataReader, self).__init__()
		self.ticker = ticker
		self.frames = {}
		self.start = datetime.datetime(2000,1,1)	
		self.end = datetime.datetime(2019,6,30)
		self.minDate = datetime.datetime(2000,1,1)

	def readData(self):
		for i in self.ticker:
			df = pdr.get_data_yahoo(i,start=self.start,end=self.end)
			#Most recently avaiable stock data
			self.frames[i] = df
			if (df.index[0] > self.minDate):
				self.minDate = df.index[0]

	def clipData(self):
		for i in self.ticker:
			self.frames[i] = self.frames[i][self.frames[i].index > self.minDate]

	def calculateReturn(self):
		for i in self.ticker:
			self.frames[i]['Returns'] = self.frames[i].Close.pct_change()

	def mergeData(self):
		self.data = pd.DataFrame()
		self.data['Date'] = self.frames[self.ticker[0]].index
		for i in self.ticker: 
			self.data[i] = self.frames[i].Returns.values
		print(self.data.head())

	def saveData(self):
		self.data.dropna(inplace=True)
		self.data.to_csv('../data/merged_data.csv',index=False)

if __name__ == '__main__':
	ticker = ['AAPL','V']
	test = DataReader(ticker)
	test.readData()
	test.clipData()
	test.calculateReturn()
	test.mergeData()
	test.saveData()