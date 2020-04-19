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
		l.append(column_name)
		data_frames[name].rename(columns={'Date':column_name}, inplace = True)
		data_frames[name].rename(columns={'Price':column_name + ' Price'}, inplace = True)
		combined = pd.concat([combined,data_frames[name].drop(columns=['Open', 'High', 'Low', 'Vol.', 'Change %'])],axis=1,sort=False)


	l.remove('BHEL_Historical_Data')
	combined.drop(columns=l,inplace=True)
	combined.rename(columns={'BHEL_Historical_Data':'Date'},inplace = True)
	combined.to_csv('../data/combined.csv',index = False)

	return combined

if __name__ == '__main__':
	path = '../data/daily_data'
	files = getFilesName(path)
	data_frames = readFiles(files)	
	combined = combineFiles(data_frames)

