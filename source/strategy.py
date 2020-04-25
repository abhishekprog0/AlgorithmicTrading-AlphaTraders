# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-04-25 13:57:44
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-04-25 16:36:15


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



def executeTrade(data,close_prices):
	prev_weights = data['old'].detach().numpy().reshape(-1)
	new_weights = data['new'].detach().numpy().reshape(-1)
	wealth = data['wealth']
	notional = (new_weights - prev_weights) * wealth 
	num_trade = notional / close_prices
	num_trade = num_trade.astype(int)



#unit test
if __name__ == '__main__':
	getData('x')
	