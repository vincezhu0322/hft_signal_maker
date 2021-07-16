from team_api import date_util as du
from team_api.data.data_getter import api
import logbook
import pandas as pd
import numba


pd.options.display.max_columns = 20


@numba.njit
def _numba_ts_align(ts: int, freq_second: int):
    second = ts % 100
    minute = (ts // 100) % 100
    hour = ts // 10000
    if second > 60 - freq_second:
        minute, second = minute + 1, 0
    else:
        second = (second - 1) // freq_second + 1
    if minute >= 60:
        hour, minute = hour + 1, 0
    return hour * 10000 + minute * 100 + second


def _time_flag(freq):
    if freq.endswith('s'):
        resample_second = int(freq.replace('second', '').replace('s', ''))
    elif freq.endswith('min') or freq.endswith('m'):
        resample_second = int(freq.replace('min', '').replace('m', '')) * 60
    else:
        resample_second = 3
    return resample_second


class HftContext:

    def __init__(self):
        self.ds = None
        self.trans_data = None
        self.order_data = None
        self.snap_data = None
        self.result_data = []

    def get_trans(self, dimension_fix=True, time_flag_freq='1min', exclude_auction=False, exclude_cancel=True):
        """
        获取逐笔成交数据

        :param dimension_fix: 是否要修正价格数据量纲，默认True
        :param time_flag_freq: 数据中'time_flag'字段的采样频次，默认为1min，可选3s/10s/1min/10min/30min
        :param exclude_auction: 是否剔除集合竞价阶段数据
        :param exclude_cancel: 是否剔除取消单
        :return: cudf.DataFrame
        """
        assert self.trans_data is not None
        res = self.trans_data.copy()
        step = _time_flag(time_flag_freq)
        res['time_flag'] = res['time'].map(lambda x: _numba_ts_align(x // 1000, step))
        if dimension_fix:
            res['price'] = res['price'] / 10000
        if exclude_auction:
            res = res[res['time'] >= 93000000]
        if exclude_cancel:
            res = res[res['transType'] != 0]
        return res

    def get_snap(self, dimension_fix=True, time_flag_freq='3s', only_trade_time=False, exclude_auction=False,
                 exclude_post_trading=False, fill_time_flag=True):
        """
        获取快照截面数据

        :param fill_time_flag: 填充缺失的时间bar
        :param dimension_fix: 是否要修正价格数据量纲，默认True
        :param time_flag_freq: 数据中'time_flag'字段的采样频次，默认为1min，可选3s/10s/1min/10min/30min
        :param only_trade_time: 是否只包含交易时间数据，默认为False
        :param exclude_auction: 是否剔除集合竞价数据，默认为False
        :param exclude_post_trading: 是否剔除盘后数据，默认为False
        :return:
        """
        assert self.snap_data is not None
        res = self.snap_data.copy()
        step = _time_flag(time_flag_freq)
        res['time_flag'] = res['time'].map(lambda x: _numba_ts_align(x // 1000, step))
        res = res.sort_values('time').drop_duplicates(subset=['code', 'time_flag'], keep='last')
        if dimension_fix:
            fix_fields = ['last', 'min', 'max', 'open', 'high', 'low']
            for i in range(1, 11):
                fix_fields.append(f'ask{i}')
                fix_fields.append(f'bid{i}')
            for field in fix_fields:
                res[field] = res[field] / 10000
        if exclude_auction:
            res = res[res['time'] >= 93000000]
        if exclude_post_trading:
            res = res[res['time'] <= 150000000]
        if only_trade_time:
            res = res[(res['time'] >= 93000000) & (res['time'] <= 113000000)
                      | (res['time'] >= 130000000) & (res['time'] <= 150000000)]
        if fill_time_flag:
            if step < 60:
                time = [int(f"{hour}{minute if minute >= 10 else f'0{minute}'}{second if second >= 10 else f'0{second}'}")
                        for hour in range(9, 16) for minute in range(60) for second in range(0, 60, step)]
            else:
                time = [int(f"{hour}{minute if minute >= 10 else f'0{minute}'}00") for hour in range(9, 16) for minute
                        in range(0, 60, int(step/60))]
            time = [t for t in time if 93000 <= t <= 150000]
            code = [c for c in set(res.code.values) for i in range(len(time))]
            time = time * len(set(res.code.values))
            temp = cudf.DataFrame(time, code).reset_index()
            temp.columns = ['code', 'time_flag']
            temp.merge(res, on=['code', 'time_flag'], how='left')
            temp.fillna(method='ffill')
        return res

    def get_snap_tensor(self, dimension_fix=True, time_flag_freq='3s', only_trade_time=False, exclude_auction=False,
                        exclude_post_trading=False, ):
        """
        获取快照截面tensor数据

    def get_snap(self, with_minute_flag=False, exclude_auction=False,):
        if with_minute_flag:
            self.snap_data['minute_flag'] = self.snap_data['time'].map(lambda x: (x - 1) // 100000 + 1)
        if exclude_auction:
            self.snap_data = self.snap_data[self.snap_data['time'] >= 93000000]
        return self.snap_data
        :param dimension_fix: 是否要修正价格数据量纲，默认True
        :param time_flag_freq: 数据中'time_flag'字段的采样频次，默认为1min，可选3s/10s/1min/10min/30min
        :param only_trade_time: 是否只包含交易时间数据，默认为False
        :param exclude_auction: 是否剔除集合竞价数据，默认为False
        :param exclude_post_trading: 是否剔除盘后数据，默认为False

        :return: (截面tensor => cupy.Array, {'codes': 股票列表, 'fields': 字段列表, 'times': 时间戳列表} => dict)
        """
        import cupy as cp
        snap = self.get_snap(dimension_fix=dimension_fix, time_flag_freq=time_flag_freq, only_trade_time=only_trade_time,
                             exclude_auction=exclude_auction, exclude_post_trading=exclude_post_trading).set_index(
            ['time_flag', 'code'])
        columns = [c for c in snap.columns if c not in ('ds', 'code', 'time_flag')]
        times = None
        codes = None
        snap_unstack = snap.unstack()
        tensor_result = []
        for c in columns:
            tensor_result.append(cp.asarray(snap_unstack[c].as_gpu_matrix()))
            if times is None:
                times = [f'{self.ds} {t//10000:02d}:{t//100%100:02d}:{t%100:02d}' for t in snap_unstack[c].index.to_pandas()]
            if codes is None:
                codes = list(snap_unstack[c].columns)
        tensor_result = cp.stack(tensor_result)
        return tensor_result, {'fields': columns, 'codes': codes, 'times': times}

    def get_order(self):
        pass

    def update_ds(self, ds):
        self.ds = ds
        self.result_data = []


class HftPipeline:

    def __init__(self, include_trans=False, include_order=False, include_snap=False,
                 minute_flag='1min'):
        self.include_trans = include_trans
        self.include_order = include_order
        self.include_snap = include_snap
        self.minute_flag = minute_flag
        self._steps = []

    def compute(self, start_ds, end_ds, universe='StockA', n_code_block=1):
        """
        计算当前高频因子值

        :param start_ds: 开始时间
        :param end_ds: 结束时间
        :param universe: 股票池
        :param n_code_block: 进行block计算时，code切分的数量
        :return:
        """
        from dataapi.stock.cudf_market import get_cudf_transaction, get_cudf_snapshot, get_cudf_order_sh, \
            get_cudf_order_sz
        import cudf
        assert n_code_block == 1, NotImplementedError()
        trading_days = du.get_between_date(start_ds, end_ds)
        context = HftContext()
        result = []
        for ds in trading_days:
            logbook.info(f'start to compute {ds}')
            context.update_ds(ds=ds)
            if isinstance(universe, list):
                code_list = universe
            else:
                code_list = list(api.universe(ds, universe).reset_index()['code'].unique())
            if self.include_snap:
                logbook.info(f'start load snap of {ds}')
                snap = get_cudf_snapshot(ds, code=code_list, source='rough_merge_v2', ns_time=False)
                context.snap_data = snap
            if self.include_trans:
                trans = get_cudf_transaction(ds, code=code_list, source='rough_merge_v0', ns_time=False)
                context.trans_data = trans
            if self.include_order:
                raise NotImplementedError()
            for step in self._steps:
                func_type, func, kwargs = step
                if func_type == 'block':
                    res = func(context)
                    res = res.reset_index()
                    res['ds'] = ds
                    context.result_data.append(res.set_index(['ds', 'code', 'time_flag']))
                else:
                    raise NotImplementedError()
            ds_res = cudf.concat(context.result_data).sort_index().to_pandas()
            result.append(ds_res)
            logbook.info(f'finish compute {ds}')
        return pd.concat(result)

    def add_block_step(self, func, **kwargs):
        """
        新增block步骤，block步骤为定义在一组code及特定时间范围内的聚合运算

        :param func: HftContext => cudf.DataFrame(index with (code, time_freq))。计算当前block对应的对齐数据
        :param kwargs:
        :return:
        """
        self._steps.append(('block', func, kwargs))

    def add_cross_step(self, func, **kwargs):
        self._steps.append(('cross', func, kwargs))


if __name__ == '__main__':
    pass


