import cudf
from hft_signal_maker.hft_pipeline import HftPipeline


def calculate_minute_bar(cxt):
    trans = cxt.get_trans(time_flag_freq='1min', only_trade_time=True)
    trans['amount'] = trans['volume'] * trans['price']
    items = trans.groupby(['code', 'time_flag']).agg({'price': ['max', 'min'], 'volume': ['sum'], 'amount': ['sum']})
    open = trans.sort_values(['time']).drop_duplicates(subset=['code', 'time_flag'], keep='first').set_index(
        ['code', 'time_flag'])['price']
    close = trans.sort_values(['time']).drop_duplicates(subset=['code', 'time_flag'], keep='last').set_index(
        ['code', 'time_flag'])['price']
    res = cudf.concat([items['price']['max'].rename('high'),
                       items['price']['min'].rename('low'),
                       items['volume']['sum'].rename('volume'),
                       items['amount']['sum'].rename('amount'),
                       open.rename('open'),
                       close.rename('close')], axis=1)
    res['vwap'] = res['amount'] / res['volume']
    return res


pipeline = HftPipeline('1min_trans_basic_bar', include_trans=True)
pipeline.add_block_step(calculate_minute_bar)
pipeline.gen_factors(['open', 'close', 'high', 'low', 'volume', 'amount', 'vwap'])


if __name__ == '__main__':
    # result = pipeline.compute(start_ds='20210608', end_ds='20210608', universe=['000001.SZ', '000637.SZ'])
    result = pipeline.compute(start_ds='20200101', end_ds='20200104', universe='StockA', n_blocks=8)
    print(result)


