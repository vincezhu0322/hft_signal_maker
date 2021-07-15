import cudf
from hft_pipeline import HftPipeline
import numpy as np
from cuml.linear_model import LinearRegression


def calculate_bsflag_regression(cxt):
    trans = cxt.get_trans(time_flag_freq='1min', exclude_auction=False, exclude_cancel=True)
    trans = trans.sort_values(['ds', 'code', 'time_flag']).reset_index(drop=True)
    trans['preprice'] = trans.groupby(['ds', 'code', 'time_flag']).price.shift(1).reset_index(drop=True)
    trans['delta_price'] = trans.price - trans.preprice
    trans.dropna(inplace=True)
    trans_group = trans.groupby(['ds', 'code', 'time_flag'])
    L_dict = {'ds': [], 'code': [], 'minute': [], 'lambda': []}
    for key, value in trans_group:
        model = LinearRegression(fit_intercept=False)
        if value.shape[0] > 1:
            x = value.bsFlag.astype('float').to_array()
            y = value.delta_price.to_array()
            model.fit(x, y)
            L_dict['ds'].append(key[0])
            L_dict['code'].append(key[1])
            L_dict['minute'].append(key[2])
            L_dict['lambda'].append(model.coef_[0])
        else:
            L_dict['ds'].append(key[0])
            L_dict['code'].append(key[1])
            L_dict['minute'].append(key[2])
            L_dict['lambda'].append(0)
    L = cudf.DataFrame(L_dict)
    return L


def calculate_bsflag_2diff_regression(cxt):
    trans = cxt.get_trans(time_flag_freq='1min', exclude_auction=False, exclude_cancel=True)
    trans = trans.sort_values(['ds', 'code', 'time_flag']).reset_index(drop=True)
    trans['preprice'] = trans.groupby(['ds', 'code', 'time_flag']).price.shift(1).reset_index(drop=True)
    trans['delta_price'] = trans.price - trans.preprice
    trans['prebsFlag'] = trans.groupby(['ds', 'code', 'time_flag']).bsFlag.shift(1).reset_index(drop=True)
    trans['delta_bsFlag'] = trans.bsFlag - trans.prebsFlag
    trans.dropna(inplace=True)
    trans_group = trans.groupby(['ds', 'code', 'time_flag'])
    L_dict = {'ds': [], 'code': [], 'minute': [], 'lambda': [], 'gamma': []}
    for key, value in trans_group:
        model = LinearRegression(fit_intercept=False)
        if value.shape[0] > 1:
            x = value[['bsFlag', 'delta_bsFlag']]
            y = value.delta_price
            model.fit(x.astype('float'), y)
            L_dict['ds'].append(key[0])
            L_dict['code'].append(key[1])
            L_dict['minute'].append(key[2])
            L_dict['lambda'].append(model.coef_[0])
            L_dict['gamma'].append(model.coef_[1])
        else:
            L_dict['ds'].append(key[0])
            L_dict['code'].append(key[1])
            L_dict['minute'].append(key[2])
            L_dict['lambda'].append(0)
            L_dict['gamma'].append(0)
    L = cudf.DataFrame(L_dict)
    return L


pipeline = HftPipeline(include_trans=True)
pipeline.add_block_step(calculate_bsflag_regression)
