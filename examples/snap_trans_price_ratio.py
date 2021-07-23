import cudf
from hft_signal_maker.hft_pipeline import HftPipeline


def calculate_snap_trans_price_ratio(cxt):
    trans = cxt.get_trans(time_flag_freq='1min', only_trade_time=True)
    trans = trans[['ds', 'code', 'time', 'bsFlag', 'price']]
    snapshot = cxt.get_snap(time_flag_freq='1min', only_trade_time=True)
    snapshot = snapshot[['ds', 'code', 'time', 'bid1', 'ask1']]
    snapshot.drop_duplicates(subset=['ds', 'code', 'time_flag'], keep='last', inplace=True)
    snapshot['price'] = (snapshot.bid1 + snapshot.ask1)
    trans.drop_duplicates(subset=['ds', 'code', 'time_flag'], keep='first', inplace=True)
    snapshot.drop(columns=['bid1', 'ask1'], inplace=True)
    snapshot = snapshot.merge(trans, on=['ds', 'code', 'time_flag'])
    snapshot['indic'] = snapshot.bsFlag * (snapshot.price_y - snapshot.price_x) / snapshot.price_x
    snapshot['indic'] = snapshot.bsFlag * (snapshot.price_y - snapshot.price_x) / snapshot.price_x
    snapshot = snapshot.drop(columns=['time_x', 'time_y']).rename({'price_x': 'snap_price', 'price_y': 'trans_price'},
                                                                  axis=1)
    return snapshot


if __name__ == '__main__':
    pipeline = HftPipeline('1min_snap_price_ratio', include_trans=True, include_snap=True)
    pipeline.add_block_step(calculate_snap_trans_price_ratio)
    data = pipeline.compute('20210101', '20210201')
