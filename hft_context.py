from hft_signal_maker.numba_util import _ts_align, get_cudf_from_arrow


class HftContext:

    def __init__(self):
        self._current_step = None
        self.ds = None
        self.trans_data = None
        self.order_data = None
        self.snap_data = None
        self.trans_wiz_order_data = None
        self._frame_data = None
        self._current_interval = (None, None)
        self.all_intervals = set()

    def get_trans(self, dimension_fix=True, time_flag_freq='1min', only_trade_time=False, exclude_auction=False,
                  exclude_cancel=True):
        """
        获取逐笔成交数据，只可在block运算中使用

        :param dimension_fix: 是否要修正价格数据量纲，默认True
        :param time_flag_freq: 数据中'time_flag'字段的采样频次，默认为1min，可选3s/10s/1min/10min/30min
        :param only_trade_time: 是否只包含交易时间数据，默认为False
        :param exclude_auction: 是否剔除集合竞价阶段数据
        :param exclude_cancel: 是否剔除取消单
        :return: cudf.DataFrame
        """
        import cudf
        assert self.trans_data is not None and self._current_step == 'block'
        res = cudf.DataFrame.from_arrow(self.trans_data[self._current_interval])
        step = _time_flag(time_flag_freq)
        res = get_cudf_from_arrow(self.trans_data[self._current_interval], step)
        if dimension_fix:
            res['price'] = res['price'] / 10000
        if only_trade_time:
            res = res[(res['time'] >= 93000000) & (res['time'] <= 113000000)
                      | (res['time'] >= 130000000) & (res['time'] <= 150000000)]
        if exclude_auction:
            res = res[res['time'] >= 93000000]
        if exclude_cancel:
            res = res[res['transType'] != 0]
        return res.sort_values(['code', 'time'])

    def get_snap(self, dimension_fix=True, time_flag_freq='3s', only_trade_time=False, exclude_auction=False,
                 exclude_post_trading=False, fill_time_flag=True):
        """
        获取快照截面数据，只可在block运算中使用

        :param fill_time_flag: 填充缺失的时间bar
        :param dimension_fix: 是否要修正价格数据量纲，默认True
        :param time_flag_freq: 数据中'time_flag'字段的采样频次，默认为1min，可选3s/10s/1min/10min/30min
        :param only_trade_time: 是否只包含交易时间数据，默认为False
        :param exclude_auction: 是否剔除集合竞价数据，默认为False
        :param exclude_post_trading: 是否剔除盘后数据，默认为False
        :return:
        """
        import cudf
        assert self.snap_data is not None and self._current_step == 'block'
        res = cudf.DataFrame.from_arrow(self.snap_data[self._current_interval])
        step = _time_flag(time_flag_freq)
        res['time_flag'] = res['time'].map(lambda x: _ts_align(x // 1000, step))
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
            scale = (step//60)*100
            if self._current_interval[0] and self._current_interval[1]:
                starttime = self._current_interval[0] + scale
                endtime = self._current_interval[1] + scale
            elif self._current_interval[0]:
                starttime = self._current_interval[0] + scale
                endtime = 150000
            elif self._current_interval[1]:
                starttime = 93000
                endtime = self._current_interval[1] + scale
            else:
                starttime = 93000
                endtime = 150000
            time = [t for t in time if starttime <= t <= endtime]
            code = [c for c in res.code.unique().to_array() for i in range(len(time))]
            time = time * len(res.code.unique().to_array())
            temp = cudf.DataFrame(time, code).reset_index()
            temp.columns = ['code', 'time_flag']
            temp = temp.merge(res, on=['code', 'time_flag'], how='left').sort_values(['code', 'time_flag'])
            temp[(temp.time_flag == starttime) | (temp.time_flag == endtime)] = temp[
                (temp.time_flag == starttime) | (temp.time_flag == endtime)].fillna(0)
            temp = temp.fillna(method='ffill')
            res = temp
        return res

    def get_snap_tensor(self, dimension_fix=True, time_flag_freq='3s', only_trade_time=False, exclude_auction=False,
                        exclude_post_trading=False, ):
        """
        获取快照截面tensor数据，只可以在block运算中使用

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

    def get_order(self, dimension_fix=True, time_flag_freq='1min', only_trade_time=False, exclude_auction=False,
                  exclude_cancel=True):
        """
        获取逐笔委托数据，只可在block运算中使用

        :param dimension_fix: 是否要修正价格数据量纲，默认True
        :param time_flag_freq: 数据中'time_flag'字段的采样频次，默认为1min，可选3s/10s/1min/10min/30min
        :param only_trade_time: 是否只包含交易时间数据，默认为False
        :param exclude_auction: 是否剔除集合竞价阶段数据
        :param exclude_cancel: 是否剔除取消单
        :return: cudf.DataFrame
        """
        import cudf
        assert self.order_data is not None and self._current_step == 'block'
        res = cudf.DataFrame.from_arrow(self.order_data[self._current_interval])
        step = _time_flag(time_flag_freq)
        res['time_flag'] = res['time'].map(lambda x: _ts_align(x // 1000, step))
        if dimension_fix:
            res['price'] = res['price'] / 10000
        if only_trade_time:
            res = res[(res['time'] >= 93000000) & (res['time'] <= 113000000)
                      | (res['time'] >= 130000000) & (res['time'] <= 150000000)]
        if exclude_auction:
            res = res[res['time'] >= 93000000]
        if exclude_cancel:
            res = res[res['ordertype'] != 4]
        return res.sort_values(['code', 'time'])

    def _update_ds(self, ds):
        self.ds = ds
        self._frame_data = None

    def get_data(self):
        """
        获取前序步骤中已经完成的计算结果，可在cross和block过程中使用

        :return: cudf.DataFrame，前序过程中计算完成的数据
        """
        return self._frame_data

    def _add_snapshot_blocks(self, snapshot_blocks):
        self.snap_data = snapshot_blocks
        self.all_intervals |= set(snapshot_blocks.keys())

    def _add_trans_blocks(self, trans_blocks):
        self.trans_data = trans_blocks
        self.all_intervals |= set(trans_blocks.keys())

    def _add_order_blocks(self, order_blocks):
        self.order_data = order_blocks
        self.all_intervals |= set(order_blocks.keys())

    def _add_trans_wiz_order_blocks(self, trans_wiz_order_blocks):
        self.trans_wiz_order_data = trans_wiz_order_blocks
        self.all_intervals |= set(trans_wiz_order_blocks.keys())

    def _update_current_interval(self, interval):
        self._current_interval = interval

    def _update_step(self, step, interval=None):
        self._current_step = step
        self._current_interval = interval


def _time_flag(freq):
    if freq.endswith('s'):
        resample_second = int(freq.replace('second', '').replace('s', ''))
    elif freq.endswith('min') or freq.endswith('m'):
        resample_second = int(freq.replace('min', '').replace('m', '')) * 60
    else:
        resample_second = 3
    return resample_second


