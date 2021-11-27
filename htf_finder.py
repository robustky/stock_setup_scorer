import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from statistics import mean
from datetime import datetime, timedelta, date
import yfinance as yf
#from stock_scoring_funcs import *
from mbs_scorer import *


cwd = os.getcwd()

# ticker_data = pd.read_csv("stock_list_feedback.csv",index_col=False) ---- for ticker in ticker_data.ticker:
TICKER_HTF_TRIGGED = []
TICKER_HTF_SCORES = []
#TICKERS = ['KEYS', 'LULU', 'TX', 'UNIT', 'NKE', 'ARE', 'LLY', 'OTIS', 'CUBE', 'LNT', 'TRMB', 'CARR', 'XM', 'DE', 'WAB', 'ROK', 'GTLS', 'RELX', 'KAR', 'NAVI', 'LSPD', 'FIVE', 'TBIO', 'CVNA', 'SIC', 'LC']
TICKERS = ['COP','KEYS']


cnt = 0

if __name__ == '__main__':
    for ticker in TICKERS:
        print("\nCount: {0} Checking the Ticker: {1}".format(cnt, ticker), end =" ")
        dummy, gl_ticker_lin_score, gl_htf_final_day_score = stock_score(ticker)
        if not np.isnan(gl_htf_final_day_score):
            print("--> HTF_score: {0:.2f}".format(gl_htf_final_day_score), end=" ")
            TICKER_HTF_TRIGGED.append(ticker)
            TICKER_HTF_SCORES.append(gl_htf_final_day_score)
        cnt += 1

    with open('htf_tickers.txt', 'w') as f:
        for i in range(len(TICKER_HTF_TRIGGED)):
            res_str = TICKER_HTF_TRIGGED[i] + ',' + TICKER_HTF_SCORES[i].astype(str) + '\n'
            f.write(res_str)
        f.close()