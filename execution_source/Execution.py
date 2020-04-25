# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-04-25 13:06:10
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-04-25 13:11:06
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as skl

from vwap_profile_1 import VWAP
import simtools as st
import vwap2 as vw

if __name__ == '__main__':
	#Trade File should be ZIP and have Columns as - 
	#'DATE', 'TIME_M', 'SYM_ROOT', 'SYM_SUFFIX', 'SIZE', 'PRICE'
	trade_path = 'e4733b5baa1d0556_csv.zip'
	#Quote File should be ZIP and have Columns as - 
	#'DATE', 'TIME_M', 'EX', 'BID', 'BIDSIZ', 'ASK', 'ASKSIZ', 'QU_COND',
	#'QU_SEQNUM', 'NATBBO_IND', 'QU_CANCEL', 'QU_SOURCE', 'SYM_ROOT','SYM_SUFFIX'
	quote_path = '938cbee848f07821_csv.zip'


	trade = pd.read_csv(trade_path)
	quote = pd.read_csv(quote_path)
	
	trade.drop(columns = ["EX"],inplace=True)
	trade.tail()

	df_quote = st.loadquotefile(quote_path)
	df_trade = st.loadtradefile(trade)

	taq_merged = st.makeTAQfile(df_trade, df_quote)


	start_time = "9:30:00" 
	end_time = "16:00:00"

	taq_merged_timed = taq_merged.between_time(start_time,end_time)
	taq_merged_timed.to_pickle('taq_merged_timed')

	taq_merged_2 = taq_merged_timed[taq_merged_timed.index.day.isin([10])]
	df_trade.to_pickle('train_data_2')


	test = VWAP('train_data_2')
	test.fitModel()
	test.plot()
	vwap_coefs = test.coef

	order_quantity = 1000
	order_side = 'b'
	results = vw.algo_loop(taq_merged_2, order_side, order_quantity, vwap_coefs,tick_coef=0)