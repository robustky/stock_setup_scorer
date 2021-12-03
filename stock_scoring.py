import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from statistics import mean
import yfinance as yf

pd.options.mode.chained_assignment = None
cwd = os.getcwd()

TICKERS = ['ALLK', 'ATRI', 'CAMT', 'CARA', 'CCRN',
           'CKPT', 'DSPG', 'ENOB', 'FULC', 'GALT']

# Setting state parameters
PLOT_STATE = True  # To see plots make it True
PRINT_STATE = True  # To see text prints make it True

# Initializing global result frame and score
FINAL_RESULT_FRAME = pd.DataFrame()  # To store result data
SCORES = []  # To store ticker and corresponding linearity score


def get_ticker_dataFrame(tick, period='1y', interval='1d'):

    data_ohlc = yf.Ticker(tick)  # Get the ticker data from yfinance
    dataFrame = data_ohlc.history(period=period, interval=interval)
    # dataFrame = dataFrame.iloc[:-49 , :] to analyze previous day

    # Check whether the data can be retrieved
    if (len(dataFrame.index) != 0 and
            not dataFrame['Close'].isnull().values.any()):
        dataFrame.reset_index(level=0, inplace=True)
        cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        dataFrame.drop(dataFrame.columns.difference(cols_to_keep), axis=1)
        dataFrame.Date = dataFrame.Date.dt.date  # Drop " 00:00:00" at the end
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
            score = ((val - min_score_val) / (max_score_val - min_score_val) * (max_score - min_score) + min_score)
    else:
        if val <= max_score_val:
            score = max_score
        elif val >= min_score_val:
            score = min_score
        else:
            score = (max_score - (val - max_score_val) / (min_score_val - max_score_val) * (max_score - min_score))

    return score


def sweet_spot_scorer(val, max_score, min_score, min_cutoff_val,
                      opt_low_val, opt_high_val, max_cutoff_val):
    if val <= min_cutoff_val:
        score = min_score
    elif min_cutoff_val < val < opt_low_val:
        score = ((val - min_cutoff_val) / (opt_low_val - min_cutoff_val) * (max_score - min_score) + min_score)
    elif opt_low_val <= val <= opt_high_val:
        score = max_score
    elif opt_high_val < val < max_cutoff_val:
        score = (max_score - (val - opt_high_val) / (max_cutoff_val - opt_high_val) * (max_score - min_score))
    elif val >= max_cutoff_val:
        score = min_score
    else:  # TODO: Raise an exception
        print("Error in sweet_spot_scoring parameters")
        score = 0

    return score


def get_regions(df, sma_in, sma_out):
    df['SMA_in'] = df.Close.rolling(sma_in).mean()
    df['SMA_in_slope'] = df['SMA_in'].diff()

    df['SMA_out'] = df.Close.rolling(sma_out).mean()
    df['SMA_out_slope'] = df['SMA_out'].diff()

    min_sma_size = min(sma_in, sma_out)

    # Get indexes of positively sloped regions
    in_region = False
    index_start = []
    index_end = []
    for i in range(50, len(df)):  # Do not interested in the first 50 days
        if df.SMA_in_slope[i] > 0 and not in_region:
            in_region = True

            # Add previous positive days
            day_add_count = 0
            if i > min_sma_size:
                for j in range(min_sma_size):
                    if (df.Close[i-j] > df.Close[i-1-j] and df.Close[i-1-j] > df.Open[i-1-j]):
                        day_add_count += 1
                    else:
                        break
            index_start.append(i - day_add_count)

        elif df.SMA_out_slope[i] < 0 and in_region:
            if (df.Close[i] <= df.Open[i]):
                in_region = False

                # Discard last negative days
                day_discard_count = 0
                for j in range(min_sma_size):
                    if (df.Close[i-j] < df.Close[i-1-j] or df.Open[i-j] < df.Close[i-1-j]):
                        day_discard_count += 1
                    else:
                        break

                index_end.append(i - day_discard_count)

    if in_region:
        index_end.append(len(df) - 1)

    region_indexes = np.array(list(zip(index_start, index_end)))  # Convert to np.array

    return df, region_indexes


def filter_min_day(region_indexes, min_day_thresh):
    deleted_indexes = []
    for i in range(len(region_indexes)):
        if ((region_indexes[i][1] - region_indexes[i][0]) < (min_day_thresh - 1)):
            deleted_indexes.append(i)
    region_indexes = np.delete(region_indexes, deleted_indexes, 0)
    return region_indexes


def get_SMA_condition(df, sma_fast, sma_slow, sma_ratio_threshold):
    df['SMA_cond'] = np.where((df.Close.rolling(sma_fast).mean() / df.Close.rolling(sma_slow).mean())
                              >= sma_ratio_threshold, 1, 0)

    intrm = df['SMA_cond'].groupby((df['SMA_cond']
                                    != df['SMA_cond'].shift())
                                   .cumsum()).cumcount() + 1
    df['SMA_cond_count'] = df['SMA_cond'] * intrm  # Consecutive count

    return df


def filter_percent_increase(df, region_indexes, min_percent_increase_multiplier, min_percent_increase_lim):
    # filters min percent increase

    df['dummy1'] = df.High-df.Low
    df['dummy2'] = (df.High-df.Close.shift(1, fill_value=df.High[0])).abs()
    df['dummy3'] = (df.Low-df.Close.shift(1, fill_value=df.Low[0])).abs()
    df['atr'] = df[['dummy1', 'dummy2', 'dummy3']].values.max(1)
    df.drop(columns=['dummy1', 'dummy2', 'dummy3'], inplace=True)
    df['natr'] = df.atr/df.Close

    df['co_tightness'] = abs((df.Close-df.Open)/df.Close)
    df['hl_tightness'] = (df.High-df.Low)/df.Close
    df['d_range'] = df.High-df.Low

    deleted_indexes = []
    for i in range(len(region_indexes)):
        ris = region_indexes[i][0]
        rie = region_indexes[i][1]
        min_percent_increase = df.hl_tightness[(
            ris-20):ris].mean() * min_percent_increase_multiplier
        min_percent_increase = min(
            min_percent_increase, min_percent_increase_lim)
        perc_increase = (df.Close[rie] - df.Close[ris]) / df.Close[ris]
        if (perc_increase < min_percent_increase):  # Consecutive day treshold
            deleted_indexes.append(i)
    region_indexes = np.delete(region_indexes, deleted_indexes, 0)

    return df, region_indexes


def get_linearity_score(df, region_indexes, min_var_ratio, max_var_ratio, min_ohlc_wick_ratio, max_ohlc_wick_ratio,
                        min_gap_ratio, max_gap_ratio, min_pos_day_ratio, max_pos_day_ratio, day_length_multiplier):

    global SCORES

    region_array = np.empty((len(df), 1))
    region_array[:] = np.NaN

    for i in range(len(df)):
        for j in range(len(region_indexes)):
            if region_indexes[j][0] <= i <= region_indexes[j][1]:
                region_array[i] = j

    df["region_id"] = region_array

    # Creates region_index_list --> start_index, in between indexes, end_index
    # Creates id_list ------------> id_no,       ind_no,           , id_no
    region_index_list = []
    id_list = []
    cnt_region = 0
    for region in region_indexes:
        region_index_list.append(list(range(region[0], region[1] + 1)))
        id_list.append([cnt_region] * (region[1] - region[0] + 1))
        cnt_region += 1

    df[["Date_End", "day_length", "fitted_close", "var_ratio", "var_ratio_mean", "ohlc_wick_ratio",
        "ohlc_wick_ratio_mean", "gap_mag", "gap_ratio_mean", "pos_day_ratio",
        "var_ratio_score", "ohlc_wick_ratio_score", "gap_ratio_score", "pos_day_ratio_score",
        "day_length_multiplier", "region_lin_score", "final_lin_score", "ticker_name"]] = np.NaN

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

        var_ratio = np.absolute(np.divide(np.subtract(
            np.array(y_array), np.array(fitted_data)), np.array(y_array)))
        df.var_ratio[ind_array] = var_ratio

        var_ratio_mean = np.mean(var_ratio)
        df.var_ratio_mean[ind_array] = var_ratio_mean

        close_open_range = np.absolute(
            df.Close[ind_array] - df.Open[ind_array])
        high_low_range = df.High[ind_array] - df.Low[ind_array]
        ohlc_range = high_low_range - close_open_range
        close_open_ave = np.divide(df.Close[ind_array] + df.Open[ind_array], 2)
        ohlc_wick_ratio = np.divide(ohlc_range, close_open_ave)
        df.ohlc_wick_ratio[ind_array] = ohlc_wick_ratio

        ohlc_wick_ratio_mean = np.mean(ohlc_wick_ratio)
        df.ohlc_wick_ratio_mean[ind_array] = ohlc_wick_ratio_mean

        gap_mag = np.absolute(np.subtract(
            np.array(df.Open[ind_array[1:]]), np.array(df.Close[ind_array[:-1]])))
        df.gap_mag[ind_array[1:]] = gap_mag

        gap_ratio_mean = np.mean(
            np.divide(gap_mag, close_open_ave[1:day_length]))
        df.gap_ratio_mean[ind_array] = gap_ratio_mean

        price_delta = np.diff(df.Close[(ind_start):(ind_end + 1)])
        pos_count = np.count_nonzero(price_delta >= 0)
        neg_count = np.count_nonzero(price_delta < 0)
        pos_ratio = (pos_count + 1) / (pos_count + neg_count + 1)
        df.pos_day_ratio[ind_array] = pos_ratio

        df.Date_End[ind_array] = df.Date[ind_end]

        var_ratio_score = (max_var_ratio - var_ratio_mean) / \
            (max_var_ratio - min_var_ratio)
        var_ratio_score = np.clip(var_ratio_score, 0, 1)
        df.var_ratio_score[ind_array] = var_ratio_score

        ohlc_wick_ratio_score = (
            max_ohlc_wick_ratio - ohlc_wick_ratio_mean) / (max_ohlc_wick_ratio - min_ohlc_wick_ratio)
        ohlc_wick_ratio_score = np.clip(ohlc_wick_ratio_score, 0, 1)
        df.ohlc_wick_ratio_score[ind_array] = ohlc_wick_ratio_score

        gap_ratio_score = (max_gap_ratio - gap_ratio_mean) / \
            (max_gap_ratio - min_gap_ratio)
        gap_ratio_score = np.clip(gap_ratio_score, 0, 1)
        df.gap_ratio_score[ind_array] = gap_ratio_score

        pos_day_ratio_score = (pos_ratio - min_pos_day_ratio) / \
            (max_pos_day_ratio - min_pos_day_ratio)
        pos_day_ratio_score = np.clip(pos_day_ratio_score, 0, 1)
        df.pos_day_ratio_score[ind_array] = pos_day_ratio_score

        df.day_length_multiplier[ind_array] = day_length*day_length_multiplier

        initial_total_score = var_ratio_score + \
            ohlc_wick_ratio_score + gap_ratio_score + pos_day_ratio_score
        region_lin_score = initial_total_score * day_length * day_length_multiplier
        df.region_lin_score[ind_array] = region_lin_score

    ticker_lin_score = df.region_lin_score[region_indexes[:, 0]].sum()
    SCORES.append([ticker, ticker_lin_score])

    return df, ticker_lin_score


def htf_score_calc(df, region_indexes):

    # Flag to pole ratio scoring parameters
    length_ratio_lim = 0.5
    ftpr_score_max = 10
    ftpr_score_min = 1
    flag_to_pole_ratio_min = 0
    flag_to_pole_ratio_opt_low_lim = 0.125
    flag_to_pole_ratio_opt_up_lim = 0.375
    flag_to_pole_ratio_max = length_ratio_lim

    # Youngness of movement scoring parameters
    SMA_cond_score_max = 10
    SMA_cond_score_min = 1
    min_SMA_cond_thresh = 0
    min_opt_SMA_cond_thresh = 5
    max_opt_SMA_cond_thresh = 40
    max_SMA_cond_thresh = 80

    df[["SMA_cond_score", "pole_score_mod", "modified_pole_length", "cons_day_count",
        "ftpr_score", "ave_cons_tightness_score", "natr_score", "htf_weighted_slope_score",
        "close_to_high_score", "sp_lp_tightness_score", "htf_score"]] = np.NaN

    region_cnt = 0
    # Loop through every upward linear movement of a ticker to check for consolidation
    for rind in region_indexes:

        if PRINT_STATE:
            print("Region id: {0} :".format(region_cnt))  # , end="")

        # Region related parameters
        pole_start_ind = rind[0]
        pole_end_ind = rind[1]
        flag_start_ind = pole_end_ind + 1

        pole_start_price = df.Open[pole_start_ind]
        pole_end_price = df.Close[pole_end_ind]
        pole_increase_price = pole_end_price - pole_start_price

        lower_price_band = pole_end_price - pole_increase_price*0.20
        upper_price_band = pole_end_price + pole_increase_price*0.20

        pole_score = df.region_lin_score[pole_start_ind]

        pole_length = df.day_length[pole_start_ind]
        flag_max_check_length = math.ceil(pole_length*length_ratio_lim)
        flag_max_check_length = min(flag_max_check_length, 10)

        # Looping through to find the HTF_setup
        for ind_flag in range(pole_start_ind+5, flag_start_ind+flag_max_check_length):
            if PRINT_STATE:
                print("Checking index: {0}... : ".format(ind_flag), end="")

            if ind_flag > df.index[-1]:  # exceeding the most current day
                if PRINT_STATE:
                    print("Reached the current day, search is finished!")
                break  # break out

            if ind_flag <= pole_end_ind:
                is_in_pole = True
                is_in_price_band_2days = True
                max_close_regional = max(df.Close[pole_start_ind:ind_flag+1])
                is_top = False if ((df.Close[ind_flag] < max_close_regional) and
                                   (df.Close[ind_flag-1] < max_close_regional)) else True
            else:
                is_in_pole = False
                is_in_price_band_2days = True if ((lower_price_band <= df.Close[ind_flag] <= upper_price_band) or
                                                  (lower_price_band <= df.Close[ind_flag-1] <= upper_price_band)) else False
                is_top = False

            if not is_in_price_band_2days:  # Check price band condition
                if PRINT_STATE:
                    print("Out of price band 2 days in a row! Terminated! Upper lim: {0} Lower lim: {1}".format(
                        upper_price_band, lower_price_band))
                break  # break out

            if is_top:  # Check top price condition
                if PRINT_STATE:
                    print("It is top flag, continue to next day!")
                continue  # continue to next day

            if df.Close[ind_flag-4] > df.Close[ind_flag-3] > df.Close[ind_flag-2] > df.Close[ind_flag-1]:
                if PRINT_STATE:
                    print("There are down days in a row, continue to next day!")
                continue  # continue to next day

            if df.SMA_cond[ind_flag] == 0:  # Check SMA condition
                if PRINT_STATE:
                    print("SMA Cond is not satisfied, continue to next day!")
                continue    # continue to next day

            dont_close_below_this = (df.High[ind_flag]-df.Low[ind_flag])*0.30 + df.Low[ind_flag]
            # failed break out, indicating to go down
            if df.Close[ind_flag] < dont_close_below_this:
                if PRINT_STATE:
                    print("Closed below critical daily range: Close limit was {0}, continue to next day!".format(
                        dont_close_below_this))
                continue    # continue to next day

            if ((df.d_range[ind_flag-1] < df.d_range[ind_flag]) and
                    (abs(df.Close[ind_flag-1]-df.Open[ind_flag-1])*0.9 < abs(df.Close[ind_flag]-df.Open[ind_flag]))):
                if PRINT_STATE:
                    print("Immediate price action, not tighter than previous day!, continue to next day!")
                continue

            if df.Low[ind_flag] < df.Low[ind_flag-1]*0.99:
                if PRINT_STATE:
                    print("Low is lower than the yesterdays low!, continue to next day!")
                continue

            if np.where(df.hl_tightness[ind_flag-19:ind_flag+1] <= 0.01, 1, 0).sum() > 5:
                if PRINT_STATE:
                    print("Low volatality in a row, Search is terminated!")
                break

            long_period = 50
            short_period = 2
            lpi = np.arange(ind_flag-long_period, ind_flag)
            spi = np.arange(ind_flag-short_period+1, ind_flag+1)

            adr_limit = 0.015
            if df.hl_tightness[lpi].mean() < adr_limit:
                if PRINT_STATE:
                    print("ADR is {0}, below the limit {1}!".format(
                        df.hl_tightness[lpi].mean(), adr_limit))
                continue

            tightness_threshold_constant = 1.5
            max_co_price = max(df.Close[spi].max(), df.Open[spi].max())
            min_co_price = min(df.Close[spi].max(), df.Open[spi].max())
            co_tightness = (max_co_price - min_co_price) / df.Close[spi].mean()
            co_ratio = co_tightness / df.co_tightness[lpi].mean()
            tightness_flag_oc = True if (co_ratio <= tightness_threshold_constant) else False

            d_range_constant = 1.5
            d_range_ratio = (df.High[spi].max() - df.Low[spi].min()) / df.d_range[lpi].mean()
            d_range_flag = True if (d_range_ratio <= d_range_constant) else False

            failed_breakout_ratio = df.hl_tightness[lpi].mean()*0.5
            # failed break out, indicating to go down
            if ((df.High[ind_flag]-df.Close[ind_flag])/df.Close[ind_flag]) > failed_breakout_ratio:
                if PRINT_STATE:
                    print("Failed breakout: Breakout limit was {0}, continue to next day!".format(
                        failed_breakout_ratio))
                continue    # continue to next day

            if d_range_flag and tightness_flag_oc:  # HTF tightness condition

                local_price_increase = df.Close[ind_flag]-pole_start_price
                local_lower_price_band = df.Close[ind_flag] - local_price_increase*0.20
                local_upper_price_band = df.Close[ind_flag] + local_price_increase*0.20

                # to improve the estimation of the transition point from pole to flag
                cons_day_count = 0
                for day_ind in range(ind_flag, pole_start_ind, -1):
                    if (((local_lower_price_band <= df.Close[day_ind] <= local_upper_price_band) and
                            (local_lower_price_band <= df.Open[day_ind] <= local_upper_price_band)) or
                            cons_day_count == 0):
                        cons_day_count += 1
                        if cons_day_count == (ind_flag-pole_start_ind):
                            modified_pole_length = 1
                            break
                    else:
                        if (day_ind+1) > pole_end_ind:  # in consolidation
                            modified_pole_length = pole_length
                            cons_day_count = ind_flag-pole_end_ind
                            break
                        else:
                            modified_pole_length = day_ind-pole_start_ind+1
                        break

                flag_to_pole_ratio = cons_day_count/modified_pole_length

                ftpr_score = sweet_spot_scorer(flag_to_pole_ratio, ftpr_score_max, ftpr_score_min,
                                               flag_to_pole_ratio_min, flag_to_pole_ratio_opt_low_lim,
                                               flag_to_pole_ratio_opt_up_lim, flag_to_pole_ratio_max)

                if cons_day_count <= 3:
                    ftpr_score = ftpr_score*0.75

                if not is_in_pole:
                    flag_to_pole_ratio2 = (ind_flag - pole_end_ind)/pole_length
                    ftpr_score2 = sweet_spot_scorer(flag_to_pole_ratio2, ftpr_score_max, ftpr_score_min,
                                                    flag_to_pole_ratio_min, flag_to_pole_ratio_opt_low_lim,
                                                    flag_to_pole_ratio_opt_up_lim, flag_to_pole_ratio_max)
                    if cons_day_count <= 3:
                        ftpr_score2 = ftpr_score2*0.75

                    if (ftpr_score2 > ftpr_score):
                        modified_pole_length = pole_length
                        cons_day_count = ind_flag-pole_end_ind
                        flag_to_pole_ratio = flag_to_pole_ratio2
                        ftpr_score = ftpr_score2

                cons_indexes = np.arange(ind_flag-cons_day_count+1, ind_flag+1)

                ave_tightness = df.co_tightness[lpi].abs().mean()

                ave_cons_close = df.Close[cons_indexes].mean()
                ave_cons_var_ratio = mean(abs(df.Close[cons_indexes] - ave_cons_close)) / ave_cons_close
                ave_hl_band_fillment = mean((df.High[cons_indexes] - df.Low[cons_indexes]) /
                                            (local_upper_price_band - local_lower_price_band))
                ave_cons_var_ratio_score = linear_scorer(ave_cons_var_ratio, 10, 1,
                                                         ave_tightness*(0.2+0.02*cons_day_count),
                                                         ave_tightness*(0.7+0.02*cons_day_count))
                ave_hl_band_fillment_score = linear_scorer(ave_hl_band_fillment, 10, 1, 0.1, 1)
                ave_cons_tightness_score = math.sqrt(ave_cons_var_ratio_score*ave_hl_band_fillment_score)

                htf_weighted_slope = (df.Close[ind_flag]-df.Close[ind_flag-3:ind_flag].mean()) / df.Close[ind_flag]
                htf_weighted_slope_score = linear_scorer(htf_weighted_slope, 10, 5, 0, -0.015)

                SMA_cond_count = df.SMA_cond_count[ind_flag]
                SMA_start_price = df.Close[ind_flag-SMA_cond_count+1]
                SMA_end_price = df.Close[ind_flag]
                SMA_diff = SMA_end_price - SMA_start_price
                SMA_gain_rate = (SMA_diff/SMA_start_price) / SMA_cond_count
                SMA_gain_rate_score = linear_scorer(SMA_gain_rate, 10, 1, 0.01, 0.001)
                SMA_cond_count_score = sweet_spot_scorer(SMA_cond_count, SMA_cond_score_max, SMA_cond_score_min,
                                                         min_SMA_cond_thresh, min_opt_SMA_cond_thresh,
                                                         max_opt_SMA_cond_thresh, max_SMA_cond_thresh)
                SMA_cond_score = math.sqrt(SMA_cond_count_score*SMA_gain_rate_score)

                adr_score = linear_scorer(df.hl_tightness[lpi].mean(), 10, 5, 0.03, 0.015)

                close_to_high = (df.Close[ind_flag]-df.Low[ind_flag]) / (df.High[ind_flag]-df.Low[ind_flag])
                close_to_high_score = linear_scorer(close_to_high, 10, 1, 0.9, 0.3)

                d_range_ratio_score = linear_scorer(d_range_ratio/d_range_constant, 10, 5, 0.5, 1)
                co_ratio_score = linear_scorer(co_ratio, 10, 5, 0.5, 1)
                sp_lp_tightness_score = math.sqrt(d_range_ratio_score*co_ratio_score)

                if is_in_pole:
                    pole_score_mod = pole_score*0.75 * modified_pole_length/pole_length
                else:
                    pole_score_mod = pole_score*0.75
                pole_score_sat = min(pole_score_mod, 10)

                htf_score = (pole_score_sat*0.5 + ftpr_score + SMA_cond_score + ave_cons_tightness_score +
                             adr_score + htf_weighted_slope_score + close_to_high_score + sp_lp_tightness_score) / 7.5

                df.SMA_cond_score[ind_flag] = SMA_cond_score
                df.pole_score_mod[ind_flag] = pole_score_mod
                df.modified_pole_length[ind_flag] = modified_pole_length
                df.cons_day_count[ind_flag] = cons_day_count
                df.ftpr_score[ind_flag] = ftpr_score
                df.ave_cons_tightness_score[ind_flag] = ave_cons_tightness_score
                df.natr_score[ind_flag] = adr_score
                df.htf_weighted_slope_score[ind_flag] = htf_weighted_slope_score
                df.close_to_high_score[ind_flag] = close_to_high_score
                df.sp_lp_tightness_score[ind_flag] = sp_lp_tightness_score
                df.htf_score[ind_flag] = htf_score
                if PRINT_STATE:
                    print("HTF found!")

            else:
                if PRINT_STATE:
                    print("Not tight enough! Continue to next day!")

        region_cnt += 1

    return df, df.htf_score.iloc[-1]


def momentum_burst_score(df_in):
    df_inc = df_in.copy(deep=True)

    df, region_indexes = get_regions(df_inc, sma_in=4, sma_out=4)
    if len(region_indexes) != 0:  # At least one upward linear region is found
        region_indexes = filter_min_day(region_indexes=region_indexes, min_day_thresh=5)
        df, region_indexes = filter_percent_increase(
            df, region_indexes=region_indexes, min_percent_increase_multiplier=2, min_percent_increase_lim=0.05)
        df, ticker_lin_score = get_linearity_score(df, region_indexes=region_indexes,
                                                   min_var_ratio=0, max_var_ratio=0.7,
                                                   min_ohlc_wick_ratio=0, max_ohlc_wick_ratio=0.06,
                                                   min_gap_ratio=0, max_gap_ratio=0.06,
                                                   min_pos_day_ratio=0.7, max_pos_day_ratio=1,
                                                   day_length_multiplier=0.2)
    else:
        ticker_lin_score = 0

    return df, region_indexes, ticker_lin_score


def high_tight_flag_score(ticker, df_in, region_indexes):
    df = get_SMA_condition(df_in, sma_fast=7, sma_slow=65, sma_ratio_threshold=1.05)
    df, htf_final_day_score = htf_score_calc(df, region_indexes)

    return df, htf_final_day_score


def plotter(ticker, df, region_indexes, ticker_lin_score):
    x = np.arange(0, len(df))
    fig, (ax, ax2) = plt.subplots(2, figsize=(12, 8),
                                  sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    for idx, val in df.iterrows():
        color = '#2CA453'
        if val['Open'] > val['Close']:
            color = '#F04730'
        ax.plot([x[idx], x[idx]], [val['Low'], val['High']], color=color)
        ax.plot([x[idx], x[idx] - 0.1],
                [val['Open'], val['Open']], color=color)
        ax.plot([x[idx], x[idx] + 0.1],
                [val['Close'], val['Close']], color=color)

    ax2.set_xticks(x[::5])  # Take once in 5
    ax2.set_xticklabels(df.Date[::5], rotation=90)
    ax.set_xticks(x, minor=True)

    ax.set_ylabel('USD')
    ax2.set_ylabel('Volume')

    ax.xaxis.grid(color='black', linestyle='dashed', which='both', alpha=0.1)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color='black', linestyle='dashed', which='both', alpha=0.1)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    for cand in region_indexes:
        ind_end = cand[1]
        ind_start = cand[0]
        indexes = list(range(ind_start, ind_end+1))
        ax.plot(indexes, df.fitted_close[ind_start:(ind_end+1)], color='black')
        txt = "\u2B00 {0:.2f}".format(df.region_lin_score[ind_start])
        ax.text((ind_start + ind_end) / 2,
                (df.Close[ind_start] + df.Close[ind_end]) / 2, txt, weight='bold')

    for htf_ind in range(len(df)):
        if not np.isnan(df.htf_score[htf_ind]):
            ax.plot(htf_ind, df.Close[htf_ind], color="purple", marker=6, markersize=10)
            txt = "\u2691: {0:.2f}".format(df.htf_score[htf_ind])
            ax.text(htf_ind, df.Close[htf_ind], txt, weight='bold', color="purple")

    # Plot volume
    ax2.bar(x, df.Volume, color='lightgrey')
    mx = df.Volume.max() * 1.1
    # Define tick locations - 0 to max in 4 steps
    yticks_ax2 = np.arange(0, mx + 1, mx / 4)
    yticks_labels_ax2 = ['{:.2f} mi'.format(i / 1000000) for i in yticks_ax2]
    ax2.yaxis.tick_right()  # Move ticks to the left side
    # Plot y ticks / skip first and last values (0 and max)
    plt.yticks(yticks_ax2[1:-1], yticks_labels_ax2[1:-1])
    plt.ylim(0, mx)

    ax3 = ax2.twinx()
    ax3.plot(x, df.SMA_cond)

    # Title
    percent_natr = df.natr.mean()*100
    percent_ocr = mean(abs(df.Close-df.Open)/df.Close)*100
    title_string = ticker + " --> Linearity Score: " + "%.2f" % ticker_lin_score + \
        " NATR: %" + "%.2f" % percent_natr + " NOCR: %" + "%.2f" % percent_ocr
    ax.set_title(title_string, loc='left', fontsize=20)
    # no spacing between the subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.show(block=False)


def stock_score(ticker):
    is_data_valid, dataFrame = get_ticker_dataFrame(ticker)
    df_htf = pd.DataFrame()
    ticker_lin_score = np.NaN
    htf_final_day_score = np.NaN
    if is_data_valid:  # Check whether the data can be retrieved
        df_mbs, region_indexes, ticker_lin_score = momentum_burst_score(dataFrame)
        if ticker_lin_score != 0:
            df_htf, htf_final_day_score = high_tight_flag_score(ticker, df_mbs, region_indexes)

            if PLOT_STATE:  # Call plotter according to PLOT_STATE
                plotter(ticker, df_htf, region_indexes, ticker_lin_score)

    return df_htf, ticker_lin_score, htf_final_day_score


if __name__ == '__main__':
    for ticker in TICKERS:
        gl_df_htf, gl_ticker_lin_score, gl_htf_final_day_score = stock_score(ticker)

        if not gl_df_htf.empty:
            # Populate the final result frame for every ticker
            FINAL_RESULT_FRAME = FINAL_RESULT_FRAME.append(gl_df_htf)

    FINAL_RESULT_FRAME.to_excel("stock_scoring_results.xlsx", float_format="%.3f")

    if PLOT_STATE:
        input("Press [Enter] to end the script!")
