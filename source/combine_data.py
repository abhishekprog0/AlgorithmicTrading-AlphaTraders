import pandas as pd
import numpy as np
import os

def getFilesName(path):
	return os.listdir(path)

def readFiles(files):
	data_frames = {}
	for name in files:
		if name.endswith('.csv'):
			df = pd.read_csv(path+'/'+name)
			data_frames[name] = df
	return data_frames

def combineFiles(data_frames):
	combined = pd.DataFrame()
	l = []
	for name in data_frames.keys():
		column_name = name.split('.')[0]
		print(column_name)
		l.append(column_name)
		# data_frames[name].rename(columns={'Date':column_name}, inplace = True)
		Date = data_frames[name].Date
		data_frames[name].rename(columns={'Close':column_name + ' Close'}, inplace = True)
		combined = pd.concat([combined,data_frames[name].drop(columns=['Date','Open', 'High', 'Low', 'Volume', 'Adj Close','Returns'])],axis=1,sort=False)

	
	combined = pd.concat([Date.drop(columns=['Open', 'High', 'Low', 'Volume', 'Adj Close','Returns']),combined],axis=1,sort=False)
	# l.remove('SBI_Historical_Data')
	# combined.drop(columns=l,inplace=True)
	# combined.rename(columns={'SBI_Historical_Data':'Date'},inplace = True)


	combined.to_csv('back_test_2.csv',index = False)

if __name__ == '__main__':
	path = '../data'
	files = getFilesName(path)

	data_frames = readFiles(files)	

	combineFiles(data_frames)