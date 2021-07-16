from hft_signal_maker.hft_pipeline import HftPipeline, HftContext


def calculate_snap_basic_bar(cxt):
    assert isinstance(cxt, HftContext)
    snap = cxt.get_snap(dimension_fix=True, time_flag_freq='1min', only_trade_time=True)
    return snap


pipeline = HftPipeline(include_snap=True)
pipeline.add_block_step(calculate_snap_basic_bar)


if __name__ == '__main__':
    # result = pipeline.compute(start_ds='20210608', end_ds='20210608', universe=['000001.SZ', '000637.SZ'])
    result = pipeline.compute(start_ds='20200102', end_ds='20200102', universe='StockA', n_blocks=8)
    print(result)

