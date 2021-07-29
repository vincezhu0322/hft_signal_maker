import cudf
from hft_signal_maker.hft_pipeline import HftPipeline
import numpy as np


def calculate_bsflag_regression(cxt):
    from cuml.linear_model import LinearRegression
    trans = cxt.get_trans(time_flag_freq='1min', only_trade_time=True)
    trans = trans.sort_values(['ds', 'code', 'time_flag']).reset_index(drop=True)
    trans['preprice'] = trans.groupby(['ds', 'code', 'time_flag']).price.shift(1).reset_index(drop=True)
    trans['delta_price'] = trans.price - trans.preprice
    trans.dropna(inplace=True)
    trans_group = trans.groupby(['ds', 'code', 'time_flag'])
    L_dict = {'ds': [], 'code': [], 'time_flag': [], 'lambda': []}
    for key, value in trans_group:
        model = LinearRegression(fit_intercept=False)
        if value.shape[0] > 1:
            x = value.bsFlag.astype('float').to_array()
            y = value.delta_price.to_array()
            model.fit(x, y)
            L_dict['ds'].append(key[0])
            L_dict['code'].append(key[1])
            L_dict['time_flag'].append(key[2])
            L_dict['lambda'].append(model.coef_[0])
        else:
            L_dict['ds'].append(key[0])
            L_dict['code'].append(key[1])
            L_dict['time_flag'].append(key[2])
            L_dict['lambda'].append(0)
    L = cudf.DataFrame(L_dict)
    return L


def calculate_bsflag_2diff_regression(cxt):
    from cuml.linear_model import LinearRegression
    trans = cxt.get_trans(time_flag_freq='1min', only_trade_time=True)
    trans = trans.sort_values(['ds', 'code', 'time_flag']).reset_index(drop=True)
    trans['preprice'] = trans.groupby(['ds', 'code', 'time_flag']).price.shift(1).reset_index(drop=True)
    trans['delta_price'] = trans.price - trans.preprice
    trans['prebsFlag'] = trans.groupby(['ds', 'code', 'time_flag']).bsFlag.shift(1).reset_index(drop=True)
    trans['delta_bsFlag'] = trans.bsFlag - trans.prebsFlag
    trans.dropna(inplace=True)
    trans_group = trans.groupby(['ds', 'code', 'time_flag'])
    L_dict = {'ds': [], 'code': [], 'time_flag': [], 'lambda': [], 'gamma': []}
    for key, value in trans_group:
        model = LinearRegression(fit_intercept=False)
        if value.shape[0] > 1:
            x = value[['bsFlag', 'delta_bsFlag']]
            y = value.delta_price
            model.fit(x.astype('float'), y)
            L_dict['ds'].append(key[0])
            L_dict['code'].append(key[1])
            L_dict['time_flag'].append(key[2])
            L_dict['lambda'].append(model.coef_[0])
            L_dict['gamma'].append(model.coef_[1])
        else:
            L_dict['ds'].append(key[0])
            L_dict['code'].append(key[1])
            L_dict['time_flag'].append(key[2])
            L_dict['lambda'].append(0)
            L_dict['gamma'].append(0)
    L = cudf.DataFrame(L_dict)
    return L


if __name__ == '__main__':
    pipeline = HftPipeline(name='1min_bsflag_regression', include_trans=True)
    pipeline.add_block_step(calculate_bsflag_regression)
