from hft_signal_maker.hft_context import HftContext
from team_api import date_util as du
from team_api.data.data_getter import api
import logbook
import pandas as pd

pd.options.display.max_columns = 20


def _cdf_cut_to_host(cdf, n=8):
    assert n in (1, 2, 4, 8)
    res = {}
    if n == 8:
        time_slice_list = [(None, 100000), (100000, 103000), (103000, 110000), (110000, 113000), (130000, 133000),
                      (133000, 140000), (140000, 143000), (143000, None)]
    elif n == 4:
        time_slice_list = [(None, 103000), (103000, 113000), (130000, 140000), (140000, None)]
    elif n == 2:
        time_slice_list = [(None, 113000), (130000, None)]
    else:
        time_slice_list = [(None, None)]
    for start_ts, end_ts in time_slice_list:
        if start_ts is not None and end_ts is not None:
            res[(start_ts, end_ts)] = cdf[(cdf['time'] >= start_ts * 1000) & (cdf['time'] <= end_ts * 1000)].to_arrow()
        elif start_ts is None and end_ts is not None:
            res[(start_ts, end_ts)] = cdf[cdf['time'] <= end_ts * 1000].to_arrow()
        elif start_ts is not None and end_ts is None:
            res[(start_ts, end_ts)] = cdf[cdf['time'] >= start_ts * 1000].to_arrow()
        else:
            res[(start_ts, end_ts)] = cdf.to_arrow()
    return res


class HftPipeline:

    def __init__(self, name, include_trans=False, include_order=False, include_snap=False,
                 time_flag='1min'):
        self.name = name
        self.include_trans = include_trans
        self.include_order = include_order
        self.include_snap = include_snap
        self.time_flag = time_flag
        self._steps = []
        self._factors = []

    def run(self, start_ds, end_ds, universe='StockA', n_blocks=1, **kwargs):
        factor_data = self.compute(start_ds, end_ds, universe, n_blocks)
        #todo: 将计算完成的数据以h5的形式记录下来

    def compute(self, start_ds, end_ds, universe='StockA', n_blocks=1):
        """
        计算当前高频因子值

        :param start_ds: 开始时间
        :param end_ds: 结束时间
        :param universe: 股票池
        :param n_blocks: 进行block计算时，数据切分的数量，切分数据后在计算时显存占用会显著减少
        :return: 计算出的最终结果，以cudf.DataFrame形式返回
        """
        from dataapi.stock.cudf_market import get_cudf_transaction, get_cudf_snapshot
        import cudf
        assert n_blocks in (1, 2, 4, 8), ValueError('n_blocks only support 1, 2, 4, 8')
        trading_days = du.get_between_date(start_ds, end_ds)
        context = HftContext()
        df_result = []
        for ds in trading_days:
            logbook.info(f'start to compute {ds}')
            context._update_ds(ds=ds)
            if isinstance(universe, list):
                code_list = universe
            else:
                code_list = list(api.universe(ds, universe).reset_index()['code'].unique())
            if self.include_snap:
                logbook.info(f'start load snap of {ds} and cut to {n_blocks} blocks')
                context._add_snapshot_blocks(_cdf_cut_to_host(get_cudf_snapshot(
                    ds, code=code_list, source='rough_merge_v2', ns_time=False), n_blocks))
                logbook.info(f'finish load snap')
            if self.include_trans:
                logbook.info(f'start load trans of {ds} and cut to {n_blocks} blocks')
                context._add_trans_blocks(_cdf_cut_to_host(get_cudf_transaction(
                    ds, code=code_list, source='rough_merge_v0', ns_time=False), n_blocks))
                logbook.info(f'finish load trans')
            if self.include_order:
                raise NotImplementedError()
            for step in self._steps:
                func_type, func, kwargs = step
                if func_type == 'block':
                    res_data = []
                    for interval in context.all_intervals:
                        context._update_step('block', interval=interval)
                        res = func(context)
                        res = res.reset_index()
                        res['ds'] = ds
                        res_data.append(res.set_index(['ds', 'code', 'time_flag']))
                    res_data = cudf.concat(res_data).sort_index()
                    if context._frame_data is None:
                        context._frame_data = res_data
                    else:
                        context._frame_data = cudf.concat([context._frame_data, res_data], axis=1)
                elif func_type == 'cross':
                    context._update_step('cross')
                    func(context)
                else:
                    raise NotImplementedError()
            df_result.append(context._frame_data.reset_index().to_arrow())
            logbook.info(f'finish compute {ds}')
        return cudf.concat([cudf.DataFrame.from_arrow(df) for df in df_result]).set_index(
            ['ds', 'code', 'time_flag'])[self._factors]

    def add_block_step(self, func, **kwargs):
        """
        新增block步骤，block步骤为定义在一组code及特定时间范围内的聚合运算

        :param func: HftContext => cudf.DataFrame(index with (code, time_freq))。计算当前block对应的对齐数据
        :param kwargs:
        :return:
        """
        self._steps.append(('block', func, kwargs))

    def add_cross_step(self, func, **kwargs):
        """
        新增cross步骤，在cross步骤中可使用当日计算出的全部数据

        :param func: HftContext => cudf.DataFrame(index with (code, time_freq))
        :param kwargs:
        :return:
        """
        self._steps.append(('cross', func, kwargs))

    def gen_factors(self, factors):
        """
        定义最终需要输出的因子

        :param factors:
        :return:
        """
        self._factors = factors


if __name__ == '__main__':
    pass


