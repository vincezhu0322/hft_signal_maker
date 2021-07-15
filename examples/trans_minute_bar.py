import cudf
from hft_pipeline import HftPipeline


def calculate_minute_bar(cxt):
    trans = cxt.get_trans(with_minute_flag=True, exclude_auction=True, exclude_cancel=True)
    trans['amount'] = trans['volume'] * trans['price']
    items = trans.groupby(['code', 'minute_flag']).agg({'price': ['max', 'min'], 'volume': ['sum'], 'amount': ['sum']})
    open = trans.sort_values(['time']).drop_duplicates(subset=['code', 'minute_flag'], keep='first').set_index(
        ['code', 'minute_flag'])['price']
    close = trans.sort_values(['time']).drop_duplicates(subset=['code', 'minute_flag'], keep='last').set_index(
        ['code', 'minute_flag'])['price']
    res = cudf.concat([items['price']['max'].rename('high'),
                       items['price']['min'].rename('low'),
                       items['volume']['sum'].rename('volume'),
                       items['amount']['sum'].rename('amount'),
                       open.rename('open'),
                       close.rename('close')], axis=1)
    res['vwap'] = res['amount'] / res['volume']
    return res


pipeline = HftPipeline(include_trans=True, include_snap=True)
pipeline.add_block_step(calculate_minute_bar)


if __name__ == '__main__':
    # result = pipeline.compute(start_ds='20210608', end_ds='20210608', universe=['000001.SZ', '000637.SZ'])
    result = pipeline.compute(start_ds='20200101', end_ds='20200110', universe='StockA', n_code_partition=1)
    print(result)


