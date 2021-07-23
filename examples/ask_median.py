import cudf
from hft_signal_maker.hft_pipeline import HftPipeline


def calculate_ask_median(cxt):
    snap = cxt.get_snap(time_flag_freq='5min', only_trade_time=True)
    snap['mid_price'] = (snap.bid1+snap.ask1)/2
    items = snap.groupby(['code', 'time_flag']).agg({'ask1': ['median'], 'bid1': ['median'], 'bid1vol': ['median'], 'ask1vol': ['median'], 'mid_price': ['median']})
    res = cudf.concat([items['bid1']['median'].rename('bid_median'),
                       items['ask1']['median'].rename('ask_median'),
                       items['bid1vol']['median'].rename('bidsize_median'),
                       items['ask1vol']['median'].rename('asksize_median'),
                       items['mid_price']['median'].rename('mid_price_median')], axis=1)
    return res


if __name__ == '__main__':
    pipeline = HftPipeline('5min_ask_median', include_snap=True)
    pipeline.add_block_step(calculate_ask_median)
    pipeline.gen_factors(['bid_median', 'ask_median', 'bidsize_median', 'asksize_median', 'mid_price_median'])
    # result = pipeline.compute(start_ds='20210608', end_ds='20210608', universe=['000001.SZ', '000637.SZ'])
    result = pipeline.compute(start_ds='20200101', end_ds='20200104', universe='StockA', n_blocks=8)
