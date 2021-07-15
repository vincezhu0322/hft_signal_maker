import cudf
import pandas as pd
from hft_pipeline import HftPipeline
from dataapi.trial.timeflag_3s import snapshot_3s, _one_stock_3s


def calculate_delta_p_covariance(cxt, method='snap'):
    cov_dict = {'ds': [], 'code': [], 'minute': [], 'covariance': []}
    if method == 'trans':
        trans = cxt.get_trans(time_flag_freq='1min', exclude_auction=False, exclude_cancel=True)
        stock_group = trans.groupby(['ds', 'code'])
        for stock, one_stock in stock_group:
            one_stock['preprice'] = one_stock.price.shift(1)
            one_stock['delta_p'] = one_stock.price - one_stock.preprice
            one_stock['pre_delta_p'] = one_stock.delta_p.shift(1)
            one_stock = one_stock.dropna()
            minute_group = one_stock.groupby('time_flag')
            for minute, one_minute in minute_group:
                cov = one_minute.delta_p.cov(one_minute.pre_delta_p)
                cov_dict['ds'].append(stock[0])
                cov_dict['code'].append(stock[1])
                cov_dict['minute'].append(minute)
                cov_dict['covariance'].append(cov)
    elif method == 'snap':
        snapshot = cxt.get_snap(time_flag_freq='1min', exclude_auction=True, exclude_post_trading=True)
        snapshot = snapshot.to_pandas()
        snapshot3s = snapshot_3s(snapshot)
        snapshot3s['price'] = (snapshot3s.bid1 + snapshot3s.ask1) / 2
        snapshot3s['preprice'] = snapshot3s.groupby(['ds', 'code']).price.shift(1)
        snapshot3s['delta_p'] = snapshot3s.price - snapshot3s.preprice
        snapshot3s['pre_delta_p'] = snapshot3s.groupby(['ds', 'code']).delta_p.shift(1)
        snapshot3s.dropna(inplace=True)
        minute_group = snapshot3s.groupby(['ds', 'code', 'minute_flag'])
        for key in minute_group.keys():
            one_minute = minute_group.get_group(key)
            cov = one_minute.delta_p.cov(one_minute.pre_delta_p)
            cov_dict['ds'].append(key[0])
            cov_dict['code'].append(key[1])
            cov_dict['minute'].append(key[2])
            cov_dict['covariance'].append(cov)
    else:
        raise ValueError(f'improper raw data')
    delta_p_cov = cudf.DataFrame(cov_dict)
    return delta_p_cov


pipeline = HftPipeline(include_trans=True, include_snap=True)
pipeline.add_block_step(calculate_delta_p_covariance)
