import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import sys

from execution_source.simtools import log_message



matplotlib.rcParams[ 'figure.figsize' ] = ( 14, 6 )

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def vwap_target( bar_num, coefs ):
	return ( coefs[ 0 ] * bar_num +
			 coefs[ 1 ] * bar_num**2 +
			 coefs[ 2 ] * bar_num**3 +
			 coefs[ 3 ] * bar_num**4 +
			 coefs[ 4 ] * bar_num**5 )

def record_trade( trade_df, idx, trade_px, trade_qty, current_bar, trade_type ):
	print("Trade Executed")
	# print(trade_df.loc[ idx ])
	# print([ idx,trade_px, trade_qty, current_bar, trade_type ])
	trade_df.loc[ idx ] = [ trade_px, trade_qty, current_bar, trade_type ]
	return

def calc_order_quantity( raw_order_qty, round_lot, qty_remaining ):
	if raw_order_qty >= round_lot: # round to nearest lot
		return np.around( int( raw_order_qty ), int( -1 * np.log10( round_lot ) ) )
	elif qty_remaining < round_lot:
		print(qty_remaining)
		return qty_remaining 
	else:
		return -1


def calc_schedule_factor( max_behind, target_shares, quantity_filled, order_quantity ):
	quantity_behind = target_shares - quantity_filled
	schedule_factor = (quantity_behind / max_behind) 
	schedule_factor = min( 1, max( -1, schedule_factor ) )
	return schedule_factor

	
def algo_loop( trading_day, order_side, original_order_quantity, vwap_coefficients, tick_coef = 0 ):
	log_message( 'Beginning VWAP run: {:s} {:d} shares'.format(order_side, original_order_quantity) )
	round_lot = 1
	avg_spread = ( trading_day.ask_px - trading_day.bid_px ).mean()
	half_spread = avg_spread / 2
	print( "Average stock spread for sample: {:.4f}".format(avg_spread) )

	order_targets = vwap_target( np.arange( 0, 390, dtype='int64' ), vwap_coefficients ) * original_order_quantity
	order_targets[-1] = original_order_quantity
	[ last_price, last_size, bid_price, bid_size, ask_price, ask_size, volume ] = np.zeros(7)
	[ trade_count, quote_count, cumulative_volume ] = [ 0, 0, 0 ]

	fair_values = pd.Series( index=trading_day.index )
	midpoints = pd.Series( index=trading_day.index )
	schedule_factors = pd.Series( index=trading_day.index )
	tick_factors = pd.Series( index=trading_day.index )
	
	trades = pd.DataFrame( columns = [ 'price' , 'shares', 'bar', 'trade_type' ], index=trading_day.index.unique() )
	max_behind = 10
	
	# MAIN EVENT LOOP
	current_bar = 0
	current_target_shares = 0

	quantity_behind = 0

	live_order = False
	live_order_price = 0.0
	live_order_quantity = 0.0

	total_quantity_filled = 0
	quantity_remaining = original_order_quantity - total_quantity_filled
	vwap_numerator = 0.0

	total_trade_count = 0
	total_agg_count = 0
	total_pass_count = 0

	midpoint = 0.0
	fair_value = 0.0
	schedule_factor = 0.0
	schedule_coef = 1.0
	
	message_type = 0   
	tick_coef = tick_coef
	tick_window = 20
	tick_factor = 0
	tick_ema_alpha = 2 / ( tick_window + 1 )
	prev_tick = 0
	prev_price = 0
	

	log_message( 'starting main loop' )
	for index, row in trading_day.iterrows():
		time_from_open = (index - pd.Timedelta( hours = 9, minutes = 30 ))
		minutes_from_open = (time_from_open.hour * 60) + time_from_open.minute
		
		if pd.isna( row.trade_px ): # it's a quote
			if ( row.bid_px > 0 and row.bid_size > 0 ):
				bid_price = row.bid_px
				bid_size = row.bid_size * round_lot
			if ( row.ask_px > 0 and row.ask_size > 0 ):
				ask_price = row.ask_px
				ask_size = row.ask_size * round_lot
			quote_count += 1
			message_type = 'q'
		else: 
			prev_price = last_price
			last_price = row.trade_px
			last_size = row.trade_size
			trade_count += 1
			cumulative_volume += row.trade_size
			vwap_numerator += last_size * last_price
			message_type = 't'

			if live_order :
				if ( order_side == 'b' ) and ( last_price <= live_order_price ) :
					fill_size = min( live_order_quantity, last_size )
					record_trade( trades, index, live_order_price, fill_size, current_bar, 'p' )
					total_quantity_filled += fill_size
					total_pass_count += 1
	
					live_order = False
					live_order_price = 0.0
					live_order_quantity = 0.0
					quantity_behind = current_target_shares - total_quantity_filled

				if ( order_side == 's' ) and ( last_price >= live_order_price ) :
					fill_size = min( live_order_quantity, last_size )
					record_trade( trades, index, live_order_price, fill_size, current_bar, 'p' )
					total_quantity_filled += fill_size
					total_pass_count += 1

					live_order = False
					live_order_price = 0.0
					live_order_quantity = 0.0
					quantity_behind = current_target_shares - total_quantity_filled

		if minutes_from_open > current_bar:
			current_bar = minutes_from_open
			current_target_shares = order_targets[ current_bar ]
			quantity_behind = current_target_shares - total_quantity_filled
	
		if message_type == 't':
			# calc the tick
			this_tick = np.sign(last_price - prev_price)
			if this_tick == 0:
				this_tick = prev_tick
			
			# now calc the tick
			if tick_factor == 0:
				tick_factor = this_tick
			else:
				tick_factor = ( tick_ema_alpha * this_tick ) + ( 1 - tick_ema_alpha ) * tick_factor    
			
			# store the last tick
			prev_tick = this_tick
			
		# PRICING LOGIC
		new_midpoint = bid_price + ( ask_price - bid_price ) / 2
		if new_midpoint > 0:
			midpoint = new_midpoint

		schedule_factor = calc_schedule_factor( max_behind, current_target_shares, 
											   total_quantity_filled, 
											   original_order_quantity )

		if midpoint == 0:
			continue
		fair_value = midpoint + ( schedule_coef * schedule_factor * half_spread ) + ( tick_coef * tick_factor * half_spread )

		fair_values[ index ] = fair_value
		midpoints[ index ] = midpoint
		schedule_factors[ index ] = schedule_factor
		tick_factors[ index ] = tick_factor
		if order_side == 'b':
			if fair_value >= ask_price and quantity_behind > round_lot: 
				total_agg_count += 1

				new_trade_price = ask_price

				new_order_quantity = calc_order_quantity( quantity_behind, round_lot, quantity_remaining )
				
				record_trade( trades, index, new_trade_price, new_order_quantity, current_bar, 'a' )
				quantity_remaining = min( 0, quantity_remaining - new_order_quantity )
				total_quantity_filled += new_order_quantity

				live_order_quantity = 0.0
				live_order_price = 0.0
				live_order = False

			else: # we're not yet willing to cross the spread, stay passive
				if quantity_behind > round_lot:
					live_order_price = bid_price
					live_order_quantity = calc_order_quantity( quantity_behind, round_lot, quantity_remaining )
					#live_order_quantity = quantity_behind
					live_order = True

		elif order_side == 's':
			if fair_value <= bid_price and quantity_behind > round_lot:
				total_agg_count += 1

				new_trade_price = bid_price

				# now place our aggressive order: assume you can execute the full size across spread
				new_order_quantity = calc_order_quantity( quantity_behind, round_lot, quantity_remaining )
				#new_order_quantity = quantity_behind
				record_trade( trades, index, new_trade_price, new_order_quantity, current_bar, 'a' )

				# update quantity remaining
				quantity_remaining = min( 0, quantity_remaining - new_order_quantity )
				total_quantity_filled += new_order_quantity

				live_order_quantity = 0.0
				live_order_price = 0.0
				live_order = False

			else: # not yet willing to cross spread
				if quantity_behind > round_lot:
					live_order_price = ask_price
					live_order_quantity = calc_order_quantity( quantity_behind, round_lot, quantity_remaining )
					live_order = True
		else:
			# we shouldn't have got here, something is wrong with our order type
			print( 'Got an unexpected order_side value: ' + str( order_side ) )

	log_message( 'end simulation loop' )
	log_message( 'order analytics' )

	trades = trades.dropna()
	day_vwap = vwap_numerator / cumulative_volume
	avg_price = (trades[ 'price' ] * trades[ 'shares' ]).sum() / trades[ 'shares' ].sum()

	log_message( 'VWAP run complete.' )
	
	#Forcing to have exact number of trades
		# num_traded = trades['shares'].sum()
		# diff = 300000 - num_traded #Gives excess share
		# trades['shares'][-1] += diff

	return { 'midpoints' : midpoints,
			 'fair_values' : fair_values,
			 'schedule_factors' : schedule_factors,
			 'tick_factors' : tick_factors,
			 'trades' : trades,
			 'quote_count' : quote_count,
			 'day_vwap' : day_vwap,
			 'avg_price' : avg_price
		   }


