# -*- coding: utf-8 -*-
# @Author: shayaan
# @Date:   2020-02-26 10:50:49
# @Last Modified by:   shayaan
# @Last Modified time: 2020-03-27 22:00:49
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as linear_model
import datetime 


class VWAP(object):
	"""docstring for VWAP"""
	def __init__(self, file_name):
		super(VWAP, self).__init__()
		self.data = pd.read_pickle(file_name)
		self.lr = linear_model.LinearRegression(fit_intercept=False)
		self.data = self.data.between_time('09:30','16:00')
		#self.data.to_csv("test_23.csv")
		self.processData()
		#print(f"{self.data_accum.head()}")
		
	def processData(self):
		self.data['minute_bars'] = (self.data.index.hour * 60) + self.data.index.minute - ((9 * 60) + 30) + 1
		#self.data['minute_bars'] = self.data[self.data.minute_bars<=389] # trims anything beyond bin 389
		#self.data.to_csv("test23.csv")
		self.minute_bars = pd.to_numeric(self.data['minute_bars']).unique()
		#print(f"{self.data.head()}")
		self.data['accum_volume'] = self.data.groupby([self.data.index.date]).cumsum()['trade_size'] 
		self.data[ 'accum_pct' ] = self.data.groupby([self.data.index.date])['accum_volume'].transform(lambda x:x/x.iloc[-1])
		self.data_accum = self.data[["minute_bars","accum_pct"]].groupby(by="minute_bars").mean()
		self.X = pd.DataFrame({'bin': self.minute_bars, 'bin2': self.minute_bars**2, 'bin3': self.minute_bars**3, 'bin4': self.minute_bars**4,'bin5': self.minute_bars**5})
		self.y = self.data_accum['accum_pct']
		self.bins = self.minute_bars
		#can replace 'data_accum' with 'data' if you are sure the extra columns ain't being used elsewhere

	def fitModel(self):
		self.model = self.lr.fit(self.X,self.y)
		self.predictions = self.lr.predict(self.X)
		self.coef = self.lr.coef_
		print(self.lr.score(self.X,self.y))
		print('The coefficient for the VWAP model are {}'.format(self.coef))

	def vwap_target(self):
		target = self.coef[0]*self.bins + self.coef[1]*self.bins**2 + self.coef[2]*self.bins**3 + self.coef[3]*self.bins**4 + self.coef[4]*self.bins**5
		return target

	def plot(self):
		target = self.vwap_target()
		plt.plot(target)
		plt.title('VWAP Plot')
		plt.show()
	'''
	def closePlot(self):
		another_day = self.data[ '09-27-2019' ][ 'close' ].reset_index( drop = True)	
		another_day = (another_day -  np.mean(another_day))/ (np.std(another_day,ddof=1))
		target = self.vwap_target()
		target = target
		another_day = another_day
		plt.plot(target,label='VWAP')
		plt.plot(another_day,label="CLose")
		plt.title('Close price and VWAP')
		plt.show()
		plt.legend()
	'''

if __name__ == '__main__':
	test = VWAP('vwap_train.pkl')
	test.fitModel()
	test.plot()