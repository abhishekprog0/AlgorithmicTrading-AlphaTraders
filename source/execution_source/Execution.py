# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-04-25 13:06:10
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-04-29 21:51:18
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as skl

from execution_source.vwap_profile import VWAP
import execution_source.simtools as st
import execution_source.vwap2 as vw

def vwapExecute(trade_file,quote_file,order,side):
	start_time = "9:30:00"
	end_time = "16:00:00"

	trade_path = '../data/trade_data/' + trade_file
	quote_path = '../data/trade_data/' + quote_file

	trade = pd.read_csv(trade_path)
	quote = pd.read_csv(quote_path)

	trade.drop(columns=["EX", "TR_SCOND", "TR_CORR", "TR_SEQNUM", "TR_ID", "TR_SOURCE", "TR_RF"], inplace=True)

	df_quote = st.loadquotefile(quote_path)
	df_trade = st.loadtradefile(trade)


	#data frame to train vwap coefs
	train_data_2 = df_trade[df_trade.index.day.isin([8,9,10])]
	train_data_2 = train_data_2.between_time(start_time, end_time)
	train_data_2.to_pickle('train_data_2.pkl')
	

	#trade data to test
	df_trade_2 = df_trade[df_trade.index.day.isin([11])]
	taq_merged = st.makeTAQfile(df_trade_2, df_quote)
	taq_merged = taq_merged.between_time(start_time, end_time)


	test = VWAP('train_data_2.pkl')
	test.fitModel()
	vwap_coefs = test.coef

	order_quantity = order
	order_side = side
	results = vw.algo_loop(taq_merged, order_side, order_quantity, vwap_coefs,tick_coef=0.5)
	
	print("\n####################")
	print("Average Price : {}".format(results['avg_price']))
	print("Day VWAP : {}".format(results['avg_price']))
	print("Passive and Aggressive trade")
	print(results['trades'].groupby(['trade_type']).count())
	print("####################\n")