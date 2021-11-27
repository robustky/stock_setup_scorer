import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from statistics import mean
from datetime import datetime, timedelta, date
import yfinance as yf

pd.options.mode.chained_assignment = None  # default='warn'
cwd = os.getcwd()

#TICKERS = ["KEYS","LULU","TX","NKE","CBRE","ABC","CNI","LNT","XM","HOLX","CUBE"]  
TICKERS = ["LULU","XM"]#,"TX","NKE"]


###########################################
########### STATE PARAMETERS ##############
#PLOT_STATE = True  # make it True to see plots
#PRINT_STATE = True  # make it True to see text prints
########### STATE PARAMETERS ##############
###########################################


#################################################
########### INITIALIZING GLOBAL PARAMETERS ######
#FINAL_RESULT_FRAME = pd.DataFrame()  # initialize final result frame, data will be added here
#SCORES = []  # to hold ticker and corresponding score
########### INITIALIZING GLOBAL PARAMETERS ######
#################################################

def get_ticker_dataFrame(tick, period='1y', interval='1d'):
    # gets the ticker and returns pandas DataFrame version
    data_ohlc = yf.Ticker(tick) # get the Ticker data from yfinance
    dataFrame = data_ohlc.history(period=period, interval=interval) # read ohlc data with specified period and interval, interval should be "1d"
    # dataFrame = dataFrame.iloc[:-49 , :] to analyze previous day
    if (len(dataFrame.index) != 0) and (not dataFrame['Close'].isnull().values.any()): # check whether the data can be retrieved
        dataFrame.reset_index(level=0, inplace=True) # to convert index from date to 0,1,2... integer and assigning date to a column value
        #dataFrame.drop(columns=["Dividends","Stock Splits"], inplace=True)  # remove unnecesary column(s)
        cols_to_keep = ['Date','Open','High','Low','Close','Volume'] # this is better instead of removing specific unnecessary colums
        dataFrame.drop(dataFrame.columns.difference(cols_to_keep), axis=1)
        dataFrame.Date = dataFrame.Date.dt.date # to remove the time stamp of " 00:00:00" at the end of the date
        is_data_valid = True
    else:
        is_data_valid = False
    return is_data_valid, dataFrame

def linear_scorer(val, max_score, min_score, max_score_val, min_score_val):
    if max_score_val >= min_score_val:
        if val >= max_score_val:
            score = max_score
        elif val <= min_score_val:
            score = min_score
        else:
            score = (val-min_score_val)/(max_score_val-min_score_val)*(max_score-min_score)+min_score
    else:
        if val <= max_score_val:
            score = max_score
        elif val >= min_score_val:
            score = min_score
        else:
            score = max_score - (val-max_score_val)/(min_score_val-max_score_val)*(max_score-min_score)

    return score

def sweet_spot_scorer(val, max_score, min_score, min_cutoff_val, opt_low_val, opt_high_val, max_cutoff_val):
    if val <= min_cutoff_val:
        score = min_score
    elif min_cutoff_val < val < opt_low_val:
        score = (val-min_cutoff_val)/(opt_low_val-min_cutoff_val) * (max_score-min_score) + min_score
    elif opt_low_val <= val <= opt_high_val:
        score = max_score
    elif opt_high_val < val < max_cutoff_val:
        score = max_score - (val-opt_high_val)/(max_cutoff_val-opt_high_val) * (max_score-min_score)
    elif val >= max_cutoff_val:
        score = min_score
    else: # exception can be raised
        print("Error in sweet_spot_scoring parameters")
        score = 0

    return score

def get_regions(df, sma_in, sma_out):

    df['SMA_in'] = df.Close.rolling(sma_in).mean()  # calculating sma of closing prices
    df['SMA_in_slope'] = df['SMA_in'].diff()

    df['SMA_out'] = df.Close.rolling(sma_out).mean()  # calculating sma of closing prices
    df['SMA_out_slope'] = df['SMA_out'].diff()

    min_sma_size = min(sma_in, sma_out)

     # gets indexes of positively sloped regions
    in_region = False
    index_start = []
    index_end = []
    for i in range(30,len(df)): # do not interested in the first 30 days
        if df.SMA_in_slope[i] > 0 and in_region == False:
            in_region = True

            # add previous positive days
            day_add_count = 0
            if i > min_sma_size:
                for j in range(min_sma_size):
                    if (df.Close[i-j] > df.Close[i-1-j]) and ((df.Close[i-1-j] > df.Open[i-1-j])):
                        day_add_count += 1
                    else:
                        break
            index_start.append(i - day_add_count)
                
        elif df.SMA_out_slope[i] < 0 and in_region == True:
            if (df.Close[i] <= df.Open[i]):
                in_region = False

                # discard last negative days
                day_discard_count = 0
                for j in range(min_sma_size):
                    if (df.Close[i-j] < df.Close[i-1-j]) or (df.Open[i-j] < df.Close[i-1-j]):
                        day_discard_count += 1
                    else:
                        break

                index_end.append(i - day_discard_count)

    if in_region:
        index_end.append(len(df) - 1)

    region_indexes = list(zip(index_start, index_end))  # makes it a vector
    region_indexes = np.array(region_indexes)  # converts to numpy array

    return df, region_indexes

def filter_min_day(region_indexes, min_day_thresh):
    deleted_indexes = []
    for i in range(len(region_indexes)):
        if ((region_indexes[i][1] - region_indexes[i][0]) < (min_day_thresh - 1)):  # consecutive day treshold
            deleted_indexes.append(i)
    region_indexes = np.delete(region_indexes, deleted_indexes, 0)
    return region_indexes

def get_SMA_condition(df, sma_fast ,sma_slow, sma_ratio_threshold):

    df['SMA_cond'] = np.where((df.Close.rolling(sma_fast).mean() / df.Close.rolling(sma_slow).mean()) >= sma_ratio_threshold, 1 ,0) # calculating validity condition for upward movement
    df['SMA_cond_count'] = df['SMA_cond'] * (df['SMA_cond'].groupby((df['SMA_cond'] != df['SMA_cond'].shift()).cumsum()).cumcount() + 1) # consecutive count

    return df

def filter_percent_increase(df, region_indexes, min_percent_increase_multiplier, min_percent_increase_lim):
    # filters min percent increase

    df['dummy1'] = df.High-df.Low
    df['dummy2'] = (df.High-df.Close.shift(1,fill_value=df.High[0])).abs()
    df['dummy3'] = (df.Low-df.Close.shift(1,fill_value=df.Low[0])).abs()
    df['atr'] = df[['dummy1', 'dummy2', 'dummy3']].values.max(1)
    df.drop(columns=['dummy1','dummy2','dummy3'], inplace=True)
    df['natr'] = df.atr/df.Close
    #natr_mean = mean(df.natr)

    df['co_tightness'] = abs((df.Close-df.Open)/df.Close)
    # df['hl_tightness'] = (df.High-df.Low)/df.Close

    df['d_range'] = (df.High/df.Low) - 1
    deleted_indexes = []
    for i in range(len(region_indexes)):
        ris = region_indexes[i][0]
        rie = region_indexes[i][1]
        min_percent_increase = df.d_range[(ris-20):ris].mean() * min_percent_increase_multiplier
        min_percent_increase = min(min_percent_increase, min_percent_increase_lim)
        perc_increase = (df.Close[rie] - df.Close[ris]) / df.Close[ris]
        if (perc_increase < min_percent_increase):  # consecutive day treshold
            deleted_indexes.append(i)
    region_indexes = np.delete(region_indexes, deleted_indexes, 0)


    return df, region_indexes

def get_linearity_score(df, region_indexes, min_var_ratio, max_var_ratio, min_ohlc_wick_ratio, max_ohlc_wick_ratio, 
             min_gap_ratio, max_gap_ratio, min_pos_day_ratio, max_pos_day_ratio, day_length_multiplier):


    region_array = np.empty((len(df),1))
    region_array[:] = np.NaN

    for i in range(len(df)):
        for j in range(len(region_indexes)):
            if region_indexes[j][0] <= i <= region_indexes[j][1]:
                region_array[i] = j

    df["region_id"] = region_array

    # creates region_index_list --> start_index, in between indexes, end_index
    # creates id_list ------------> id_no,       ind_no,           , id_no
    region_index_list = []
    id_list = []
    cnt_region = 0
    for region in region_indexes:
        region_index_list.append(list(range(region[0], region[1] + 1)))
        id_list.append([cnt_region] * (region[1] - region[0] + 1))
        cnt_region += 1

    df[["Date_End","day_length","fitted_close","var_ratio","var_ratio_mean","ohlc_wick_ratio",
    "ohlc_wick_ratio_mean","gap_mag","gap_ratio_mean","pos_day_ratio",
    "var_ratio_score","ohlc_wick_ratio_score", "gap_ratio_score","pos_day_ratio_score","day_length_multiplier",
    "region_lin_score","final_lin_score","ticker_name"]] = np.NaN

    for cand in region_indexes:
        ind_start = cand[0]
        ind_end = cand[1]
        ind_array = np.arange(ind_start, ind_end + 1)

        day_length = ind_end - ind_start + 1
        df.day_length[ind_array] = day_length

        x_array = list(range(ind_end - ind_start + 1))
        y_array = df.Close[ind_start:(ind_end + 1)]
        slope = np.polyfit(x_array, y_array, 1)
        fitted_data = range(ind_end - ind_start + 1) * slope[0] + slope[1]
        df.fitted_close[ind_array] = fitted_data

        var_ratio = np.absolute(np.divide(np.subtract(np.array(y_array), np.array(fitted_data)), np.array(y_array)))
        df.var_ratio[ind_array] = var_ratio

        var_ratio_mean = np.mean(var_ratio)
        df.var_ratio_mean[ind_array] = var_ratio_mean

        
        close_open_range = np.absolute(df.Close[ind_array] - df.Open[ind_array])
        high_low_range = df.High[ind_array] - df.Low[ind_array]
        ohlc_range = high_low_range - close_open_range
        close_open_ave = np.divide(df.Close[ind_array] + df.Open[ind_array], 2)
        ohlc_wick_ratio = np.divide(ohlc_range, close_open_ave)
        df.ohlc_wick_ratio[ind_array] = ohlc_wick_ratio

        ohlc_wick_ratio_mean = np.mean(ohlc_wick_ratio)
        df.ohlc_wick_ratio_mean[ind_array] = ohlc_wick_ratio_mean

        gap_mag = np.absolute(np.subtract(np.array(df.Open[ind_array[1:]]), np.array(df.Close[ind_array[:-1]])))
        df.gap_mag[ind_array[1:]] = gap_mag

        gap_ratio_mean = np.mean(np.divide(gap_mag, close_open_ave[1:day_length]))
        df.gap_ratio_mean[ind_array] = gap_ratio_mean

        price_delta = np.diff(df.Close[(ind_start):(ind_end + 1)])
        pos_count = np.count_nonzero(price_delta >= 0)
        neg_count = np.count_nonzero(price_delta < 0)
        pos_ratio = (pos_count + 1) / (pos_count + neg_count + 1)
        df.pos_day_ratio[ind_array] = pos_ratio

        df.Date_End[ind_array] = df.Date[ind_end]


        var_ratio_score = (max_var_ratio - var_ratio_mean) / (max_var_ratio - min_var_ratio)
        var_ratio_score = np.clip(var_ratio_score, 0, 1)
        df.var_ratio_score[ind_array] = var_ratio_score

        ohlc_wick_ratio_score = (max_ohlc_wick_ratio - ohlc_wick_ratio_mean) / (max_ohlc_wick_ratio - min_ohlc_wick_ratio)
        ohlc_wick_ratio_score = np.clip(ohlc_wick_ratio_score, 0, 1)
        df.ohlc_wick_ratio_score[ind_array] = ohlc_wick_ratio_score

        gap_ratio_score = (max_gap_ratio - gap_ratio_mean) / (max_gap_ratio - min_gap_ratio)
        gap_ratio_score = np.clip(gap_ratio_score, 0, 1)
        df.gap_ratio_score[ind_array] = gap_ratio_score

        pos_day_ratio_score = (pos_ratio - min_pos_day_ratio) / (max_pos_day_ratio - min_pos_day_ratio)
        pos_day_ratio_score = np.clip(pos_day_ratio_score, 0, 1)
        df.pos_day_ratio_score[ind_array] = pos_day_ratio_score

        df.day_length_multiplier[ind_array] = day_length*day_length_multiplier

        initial_total_score = var_ratio_score + ohlc_wick_ratio_score + gap_ratio_score + pos_day_ratio_score
        region_lin_score = initial_total_score * day_length * day_length_multiplier
        df.region_lin_score[ind_array] = region_lin_score


    ticker_lin_score = df.region_lin_score[region_indexes[:,0]].sum()

    return df, ticker_lin_score

def htf_score_calc(df, region_indexes):

    ##########################################################
    #####START###### HTF DETECTION PARAMETERS #####START###### 

    tightness_threshold_constant = 0.75   # tightness_threshold = average_tightness * tightness_threshold_constant --> if price action is tighter than the tightness_hreshold, it is cosidered tight
    tightness_threshold_2days_constant = 1.25   # tightness_threshold = average_tightness * tightness_threshold_constant --> if price action is tighter than the tightness_hreshold, it is cosidered tight
    tightness_threshold_lower_lim_percent = 0.0075 # minimum possible tightness threshold percent

    #price_band_lower_tightness_constant = 1.5 # defining price band thickness with respect to average pole tightness
    #price_band_upper_tightness_constant = 2.5 # defining price band thickness with respect to average pole tightness
    length_ratio_lim = 0.75         # htf search area length, ratio to previous pole length (i.e: if the leg(i.e pole) length is 12 days, max htf check length is 9 days)

    #####END###### HTF DETECTION PARAMETERS #####END##########
    ##########################################################


    ########################################################
    #####START###### HTF SCORING PARAMETERS #####START###### 

    # flag to pole ratio scoring parameters
    ftpr_score_max = 10    # maximum allowable ftpr score
    ftpr_score_min = 1    # minimum allowable ftpr score, if the htf is found during the linear movement it directly gets the ftpr_score_min
    flag_to_pole_ratio_min = 0              # when ftpr ratio is between min and opt_low_lim, ftpr score is between ftpr_score_min and ftpr_score_max, score is linearly distrubuted
    flag_to_pole_ratio_opt_low_lim = 0.25   # ftpr ratio optimum lower lim --> when ftpr is between opt. upper and lower lim, ftpr score is eqaul to ftpr_score_max
    flag_to_pole_ratio_opt_up_lim = 0.5     # ftpr ratio optimum upper lim --> when ftpr is between opt. upper and lower lim, ftpr score is eqaul to ftpr_score_max
    flag_to_pole_ratio_max = length_ratio_lim # when ftpr ratio is between opt_up_lim and max, ftpr score is between ftpr_score_min and ftpr_score_max, score is linearly distrubuted

    # lets handle the youngness of movement scoring differently
    # scoring acording to SMA_cond (last consecutively day count where SMA_cond is satisfied)
    SMA_cond_score_max = 10
    SMA_cond_score_min = 1
    min_SMA_cond_thresh = 0         # days
    min_opt_SMA_cond_thresh = 5    # days
    max_opt_SMA_cond_thresh = 40    # days
    max_SMA_cond_thresh = 80        # days
                                                                                                 
    #####END###### HTF SCORING PARAMETERS #####END##########
    ########################################################

    df[["SMA_cond_score","pole_score_mod","modified_pole_length","cons_day_count",
        "ftpr_score","ave_cons_tightness_score","natr_score","htf_weighted_slope_score","htf_score"]] = np.NaN

    region_cnt = 0
    # looping through every upward linear movement of a ticker to check for consolidation (i.e pole-flag check)
    for rind in region_indexes:

        #if PRINT_STATE: print("Region id: {0} :".format(region_cnt))#, end="")

        # Region Related Parameters
        pole_start_ind = rind[0]
        pole_end_ind = rind[1]
        pind_array = np.arange(pole_start_ind, pole_end_ind + 1)
        flag_start_ind = pole_end_ind + 1

        pole_start_price = df.Open[pole_start_ind]
        pole_end_price = df.Close[pole_end_ind]
        pole_increase_price = pole_end_price - pole_start_price

        lower_price_band = pole_end_price - pole_increase_price*0.25
        upper_price_band = pole_end_price + pole_increase_price*0.20

        pole_score = df.region_lin_score[pole_start_ind]

        pole_length = df.day_length[pole_start_ind]
        flag_max_check_length = math.ceil(pole_length*length_ratio_lim)

        # Looping through to find the HTF_setup
        for ind_flag in range(pole_start_ind+5, flag_start_ind+flag_max_check_length):
            #if PRINT_STATE: print("Checking index: {0}... : ".format(ind_flag), end="")

            if ind_flag > df.index[-1]: # exceeding the most current day
                #if PRINT_STATE: print("Reached the current day, search is finished!")
                break # break out

            if ind_flag <= pole_end_ind:
                is_in_pole = True
                is_in_price_band_2days = True
                max_close_regional = max(df.Close[pole_start_ind:ind_flag+1])
                is_top = False if ((df.Close[ind_flag]<max_close_regional) and 
                                   (df.Close[ind_flag-1]<max_close_regional)) else True            
            else:
                is_in_pole = False
                is_in_price_band_2days = True if ((lower_price_band <= df.Close[ind_flag] <= upper_price_band) or 
                                                  (lower_price_band <= df.Close[ind_flag-1] <= upper_price_band)) else False
                is_top = False
            

            if not is_in_price_band_2days: # violating the price band condition
                #if PRINT_STATE: print("Out of price band 2 days in a row! Search is terminated! Upper lim: {0} Lower lim: {1}".format(upper_price_band, lower_price_band))
                break # break out

            if is_top:  # not satisfying the not being a top price condition
                #if PRINT_STATE: print("It is top flag, continue to next day!")
                continue # continue to next day

            down_day_count = np.where(df.Close[(ind_flag-5):(ind_flag+1)]-df.Open[(ind_flag-5):(ind_flag+1)] < 0, 1 ,0).sum() # downday count of last 6 days
            if down_day_count >= 5: # last days are mostly down days
                #if PRINT_STATE: print("There are down days in a row, continue to next day!")
                continue # continue to next day

            if df.SMA_cond[ind_flag] == 0: # not satisfying SMA cond
                #if PRINT_STATE: print("SMA Cond is not satisfied, continue to next day!")
                continue    # continue to next day

            dont_close_below_this = (df.High[ind_flag]-df.Low[ind_flag])*0.25 + df.Low[ind_flag]
            if df.Close[ind_flag] < dont_close_below_this: # failed break out, indicating to go down
                #if PRINT_STATE: print("Closed below critical daily range: Close limit was {0}, continue to next day!".format(dont_close_below_this))
                continue    # continue to next day

            if df.d_range[ind_flag-20:ind_flag].mean() < 0.02:
                # not ranging enough
                continue

            co_tight_mean = df.co_tightness[ind_flag-20:ind_flag].mean()
            co_tight_thresh = co_tight_mean*tightness_threshold_constant

            ave_high_low_tightness = mean(abs(df.hl_tightness))

            natr_mean = df.natr.mean()
            failed_breakout_ratio = ave_high_low_tightness*0.5
            if ((df.High[ind_flag]-df.Close[ind_flag])/df.Close[ind_flag]) > failed_breakout_ratio: # failed break out, indicating to go down
                #if PRINT_STATE: print("Failed breakout: Breakout limit was {0}, continue to next day!".format(failed_breakout_ratio))
                continue    # continue to next day


            # Calculating tightness parameters and evaluating flags to trigger HTF condition
            tightness_flag_oc = True if ((abs(df.co_tightness[ind_flag]) <= close_open_tightness_ratio_threshold) and
                                         (abs(df.co_tightness[ind_flag-1]) <= close_open_tightness_ratio_threshold)) else False # magnitude of tightness
            # tightness_flag_hl = True if ((abs(df.hl_tightness[ind_flag]) <= high_low_tightness_ratio_threshold) and
                                        #  (abs(df.hl_tightness[ind_flag-1]) <= high_low_tightness_ratio_threshold)) else False # magnitude of tightness

            max_co_price =  max(df.Close[ind_flag],df.Close[ind_flag-1],df.Open[ind_flag],df.Open[ind_flag-1])
            min_co_price =  min(df.Close[ind_flag],df.Close[ind_flag-1],df.Open[ind_flag],df.Open[ind_flag-1])
            co_2days_tightness = (max_co_price - min_co_price) / df.Close[ind_flag]
            co_2days_tightness_flag = True if co_2days_tightness <= close_open_tightness_ratio_2days_threshold else False # magnitude of tightness

            # max_hl_price = max(df.High[ind_flag],df.High[ind_flag-1])
            # min_hl_price = min(df.Low[ind_flag],df.Low[ind_flag-1])
            # hl_2days_tightness = (max_hl_price - min_hl_price) / df.Close[ind_flag]
            # hl_2days_tightness_flag = True if hl_2days_tightness <= high_low_tightness_ratio_2days_threshold else False # magnitude of tightness

            # if co_2days_tightness_flag and hl_2days_tightness_flag and tightness_flag_oc and tightness_flag_hl: # HTF tightness condition
            if co_2days_tightness_flag  and tightness_flag_oc: # HTF tightness condition


                local_price_increase = df.Close[ind_flag]-pole_start_price
                local_lower_price_band = df.Close[ind_flag] - local_price_increase*0.20
                local_upper_price_band = df.Close[ind_flag] + local_price_increase*0.20

                # to improve the estimation of the transition point from pole to flag
                cons_day_count = 0
                for day_ind in range(ind_flag, pole_start_ind, -1):
                    if ((local_lower_price_band <= df.Close[day_ind] <= local_upper_price_band) and
                        (local_lower_price_band <= df.Open[day_ind] <= local_upper_price_band)) or (cons_day_count == 0):
                        cons_day_count += 1
                        if cons_day_count == (ind_flag-pole_start_ind):
                            modified_pole_length = 1
                            ave_cons_natr = df.natr[day_ind:(ind_flag+1)].mean()
                            break
                    else:
                        if (day_ind+1) > pole_end_ind: # in consolidation
                            modified_pole_length = pole_length
                            cons_day_count = ind_flag-pole_end_ind
                            ave_cons_natr = df.natr[(pole_end_ind+1):(ind_flag+1)].mean()
                            break
                        else:
                            modified_pole_length = day_ind-pole_start_ind+1
                            ave_cons_natr = df.natr[(day_ind+1):(ind_flag+1)].mean()
                        break

                flag_to_pole_ratio = cons_day_count/modified_pole_length

                ftpr_score = sweet_spot_scorer(flag_to_pole_ratio, ftpr_score_max, ftpr_score_min,
                flag_to_pole_ratio_min, flag_to_pole_ratio_opt_low_lim, flag_to_pole_ratio_opt_up_lim, flag_to_pole_ratio_max)
                
                if not is_in_pole:
                    flag_to_pole_ratio2 = (ind_flag - pole_end_ind)/pole_length
                    ftpr_score2 = sweet_spot_scorer(flag_to_pole_ratio2, ftpr_score_max, ftpr_score_min,
                                flag_to_pole_ratio_min, flag_to_pole_ratio_opt_low_lim, flag_to_pole_ratio_opt_up_lim, flag_to_pole_ratio_max)
                
                    if (ftpr_score2 > ftpr_score):
                        modified_pole_length = pole_length
                        cons_day_count = ind_flag-pole_end_ind
                        flag_to_pole_ratio = flag_to_pole_ratio2
                        ftpr_score = ftpr_score2

                cons_indexes = np.arange(ind_flag-cons_day_count+1, ind_flag+1)
                
                #ave_cons_natr = df.natr[cons_indexes].mean()
                #ave_cons_tightness_score = linear_scorer(ave_cons_natr/natr_mean, 2, 1, 0.3, 1)

                ave_cons_close = df.Close[cons_indexes].mean()
                ave_cons_var_ratio = mean(abs(df.Close[cons_indexes] - ave_cons_close)) / ave_cons_close
                ave_hl_band_fillment = mean((df.High[cons_indexes] - df.Low[cons_indexes]) / (local_upper_price_band - local_lower_price_band))
                ave_cons_var_ratio_score = linear_scorer(ave_cons_var_ratio, 10, 1, ave_tightness*(0.25+0.01*cons_day_count), ave_tightness*(0.75+0.01*cons_day_count))
                ave_hl_band_fillment_score = linear_scorer(ave_hl_band_fillment, 10, 1, 0.1, 1)
                ave_cons_tightness_score = math.sqrt(ave_cons_var_ratio_score*ave_hl_band_fillment_score)

                #htf_weighted_slope = ((df.Close.diff()[ind_flag-2:ind_flag+1] * [0.17, 0.33, 0.50]).sum()) / df.Close[ind_flag]
                #htf_weighted_slope_score = linear_scorer(htf_weighted_slope, 1, 0.5, 0, -0.02)
                
                htf_weighted_slope = (df.Close[ind_flag]-df.Close[ind_flag-3:ind_flag].mean()) / df.Close[ind_flag]
                htf_weighted_slope_score = linear_scorer(htf_weighted_slope, 10, 5, 0, -0.015)

                SMA_cond_count = df.SMA_cond_count[ind_flag]
                SMA_start_price = df.Close[ind_flag-SMA_cond_count+1]
                SMA_end_price = df.Close[ind_flag]
                SMA_diff = SMA_end_price - SMA_start_price
                SMA_gain_rate = SMA_diff/SMA_start_price
                SMA_gain_rate_score = linear_scorer(SMA_gain_rate, 10, 1, 0.005, 0.001)
                SMA_cond_count_score = sweet_spot_scorer(SMA_cond_count, SMA_cond_score_max, SMA_cond_score_min,
                                                    min_SMA_cond_thresh, min_opt_SMA_cond_thresh, max_opt_SMA_cond_thresh, max_SMA_cond_thresh)
                SMA_cond_score = math.sqrt(SMA_cond_count_score*SMA_gain_rate_score)
                
                pole_score_mod = pole_score * modified_pole_length/pole_length # taking ratio of the length to get estimated pole_score for backtesting
                
                natr_score = linear_scorer(natr_mean, 10, 5, 0.035, 0.015)

                pole_score_sat = min(pole_score_mod, 10)

                #htf_score = ((pole_score_sat * (ftpr_score+SMA_cond_score+ave_cons_tightness_score+natr_score)**2)**(1./3))*htf_weighted_slope_score
                htf_score = (pole_score_sat*ftpr_score*SMA_cond_score*ave_cons_tightness_score*natr_score*htf_weighted_slope_score)**(1./6)


                df.SMA_cond_score[ind_flag] = SMA_cond_score
                df.pole_score_mod[ind_flag] = pole_score_mod
                df.modified_pole_length[ind_flag] = modified_pole_length
                df.cons_day_count[ind_flag] = cons_day_count
                df.ftpr_score[ind_flag] = ftpr_score
                df.ave_cons_tightness_score[ind_flag] = ave_cons_tightness_score
                df.natr_score[ind_flag] = natr_score
                df.htf_weighted_slope_score[ind_flag] = htf_weighted_slope_score
                df.htf_score[ind_flag] = htf_score
                #if PRINT_STATE: print("HTF found!")

            #else:
                #if PRINT_STATE: print("Not tight enough! Continue to next day!")

        region_cnt += 1

    return df, df.htf_score.iloc[-1]

def momentum_burst_score(df_in):
    df_inc = df_in.copy(deep=True)

    df, region_indexes = get_regions(df_inc, sma_in=4, sma_out=4)
    if len(region_indexes) != 0: # at least one upward linear region is found
        region_indexes = filter_min_day(region_indexes=region_indexes, min_day_thresh=5)
        df, region_indexes = filter_percent_increase(df, region_indexes=region_indexes, min_percent_increase_multiplier=2, min_percent_increase_lim = 0.05)
        df, ticker_lin_score = get_linearity_score(df, region_indexes=region_indexes, 
                            min_var_ratio=0,          max_var_ratio=0.7,
                            min_ohlc_wick_ratio=0,    max_ohlc_wick_ratio=0.06,
                            min_gap_ratio=0,          max_gap_ratio=0.06,
                            min_pos_day_ratio=0.7,    max_pos_day_ratio=1,
                            day_length_multiplier=0.2)
    else:
        ticker_lin_score = 0
        
    return df, region_indexes, ticker_lin_score
    
def high_tight_flag_score(ticker, df_in, region_indexes):

    df = get_SMA_condition(df_in, sma_fast=7 ,sma_slow=65, sma_ratio_threshold=1.05)
    df, htf_final_day_score = htf_score_calc(df, region_indexes)

    return df, htf_final_day_score


def plotter(ticker, df, region_indexes, ticker_lin_score):

    x = np.arange(0, len(df))
    fig, (ax, ax2) = plt.subplots(2, figsize=(12, 8), sharex = True, gridspec_kw={'height_ratios': [4, 1]})
    for idx, val in df.iterrows():
        color = '#2CA453'
        if val['Open'] > val['Close']: color = '#F04730'
        ax.plot([x[idx], x[idx]], [val['Low'], val['High']], color=color)
        ax.plot([x[idx], x[idx] - 0.1], [val['Open'], val['Open']], color=color)
        ax.plot([x[idx], x[idx] + 0.1], [val['Close'], val['Close']], color=color)

    # ticks top plot
    ax2.set_xticks(x[::5]) # take every 5 data
    #ax2.set_xticklabels(df.Date.dt.date[::5], rotation=90) # take every 5 data and rotate
    ax2.set_xticklabels(df.Date[::5], rotation=90) # take every 5 data and rotate
    ax.set_xticks(x, minor=True)
    # labels
    ax.set_ylabel('USD')
    ax2.set_ylabel('Volume')
    # grid
    ax.xaxis.grid(color='black', linestyle='dashed', which='both', alpha=0.1)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color='black', linestyle='dashed', which='both', alpha=0.1)
    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)


    for cand in region_indexes:
        ind_end = cand[1]
        ind_start = cand[0]
        indexes = list(range(ind_start,ind_end+1))
        ax.plot(indexes, df.fitted_close[ind_start:(ind_end+1)], color='black')
        txt = "\u2B00 {0:.2f}".format(df.region_lin_score[ind_start])
        ax.text((ind_start + ind_end) / 2, (df.Close[ind_start] + df.Close[ind_end]) / 2, txt, weight='bold')

    for htf_ind in range(len(df)):
        if not np.isnan(df.htf_score[htf_ind]):
            ax.plot(htf_ind, df.Close[htf_ind], color="purple", marker=6, markersize=10) # marker options: "^", 6, 10
            txt = "\u2691: {0:.2f}".format(df.htf_score[htf_ind])
            ax.text(htf_ind, df.Close[htf_ind], txt, weight='bold', color="purple")

 
    # plot volume
    ax2.bar(x, df.Volume, color='lightgrey')
    # get max volume + 10%
    mx = df.Volume.max() * 1.1
    # define tick locations - 0 to max in 4 steps
    yticks_ax2 = np.arange(0, mx + 1, mx / 4)
    # create labels for ticks. Replace 1.000.000 by 'mi'
    yticks_labels_ax2 = ['{:.2f} mi'.format(i / 1000000) for i in yticks_ax2]
    ax2.yaxis.tick_right()  # Move ticks to the left side
    # plot y ticks / skip first and last values (0 and max)
    plt.yticks(yticks_ax2[1:-1], yticks_labels_ax2[1:-1])
    plt.ylim(0, mx)

    ax3 = ax2.twinx()
    ax3.plot(x,df.SMA_cond) # check how to make 2 axes for 2 plots

    # title
    percent_natr = df.natr.mean()*100
    percent_ocr = mean(abs(df.Close-df.Open)/df.Close)*100
    title_string = ticker + " --> Linearity Score: " + "%.2f" % ticker_lin_score + " NATR: %" + "%.2f" % percent_natr + " NOCR: %" + "%.2f" % percent_ocr
    ax.set_title(title_string, loc='left', fontsize=20)
    # no spacing between the subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.show(block=False)


def stock_score(ticker):
    is_data_valid, dataFrame = get_ticker_dataFrame(ticker)
    df_htf = pd.DataFrame()   
    ticker_lin_score = np.NaN
    htf_final_day_score = np.NaN
    if is_data_valid:  # check whether the data can be retrieved
        df_mbs, region_indexes, ticker_lin_score = momentum_burst_score(dataFrame)
        if ticker_lin_score != 0: 
            df_htf, htf_final_day_score = high_tight_flag_score(ticker, df_mbs, region_indexes)

    #if PLOT_STATE == True: # calling plotter according to PLOT_STATE
        #plotter(ticker, df_htf, region_indexes, ticker_lin_score) # calling the function, passing it the ticker and dataframe

    return df_htf, ticker_lin_score, htf_final_day_score

#if __name__ == '__main__':
    #for ticker in TICKERS:
        #gl_df_htf, gl_ticker_lin_score, gl_htf_final_day_score = stock_score(ticker)

        #htf_scores, _ = high_tight_flag_score(tick, df, result_frame) # calling the function, passing it the ticker and dataframe
        #result_frame = result_frame.join(htf_scores, how='left') # joining 2 frames (i.e result_frame | htf_scores)
        #FINAL_RESULT_FRAME = FINAL_RESULT_FRAME.append(gl_df_htf) # populating the final result frame for every ticker


    #FINAL_RESULT_FRAME.to_excel("stock_scoring_results.xlsx", float_format="%.3f")

    #if PLOT_STATE == True:
        #input("Press [Enter] to end the script!")