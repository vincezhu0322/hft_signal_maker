from hft_signal_maker.hft_context import HftContext, _time_flag
from tools import date_util as du
import logbook
import pandas as pd

from iosdk.adapter.freefactors_h5 import FactorH5Writer, FactorH5Reader

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


def _read_one_day(fname, ds, time_range=None, code_list=None, factor_list=None):
    from datetime import datetime
    file_name = f'/mnt/lustre/home/zwx/hft_signal_maker/factors/{fname}/{ds}.h5'
    reader = FactorH5Reader(file_name)
    if not code_list:
        code_list = reader.code_list
    if not factor_list:
        factor_list = reader.factor_list
    if not time_range:
        unixtime_list = reader.unixtime_list
    else:
        starttime = time_range[0]
        endtime = time_range[1]
        start_h = int(starttime.split(':')[0])
        start_m = int(starttime.split(':')[1])
        start_s = int(starttime.split(':')[2])
        start = datetime(int(ds[:4]), int(ds[4:6]), int(ds[6:8]), start_h, start_m, start_s).timestamp() * 1e6
        end_h = int(endtime.split(':')[0])
        end_m = int(endtime.split(':')[1])
        end_s = int(endtime.split(':')[2])
        end = datetime(int(ds[:4]), int(ds[4:6]), int(ds[6:8]), end_h, end_m, end_s).timestamp() * 1e6
        unixtime_list = [t for t in reader.unixtime_list if t > start and t < end]
    res = reader.read(time_list=unixtime_list, code_list=code_list, factor_list=factor_list, output_type="df")
    metadata = reader.read_metadata()
    res.reset_index(inplace=True)
    res.insert(0, 'time', res.unixtime.apply(lambda x: str(datetime.fromtimestamp(x / 1e6))[-8:].replace(':', '')))
    res['time'] = res.time.astype('int32')
    res.insert(0, 'ds', ds)
    return res


class HftPipeline:

    def __init__(self, name, include_trans=False, include_order=False, include_snap=False,
                 include_trans_wiz_order=False,
                 time_flag='1min'):
        self.name = name
        self.include_trans = include_trans
        self.include_order = include_order
        self.include_snap = include_snap
        self.include_trans_wiz_order = include_trans_wiz_order
        self.time_flag = time_flag
        self._steps = []
        self._factors = []

    def run(self, start_ds, end_ds, universe='StockA', n_blocks=1, **kwargs):
        import os
        import cupy as cp
        import numpy as np
        import cudf
        factor_data = self.compute(start_ds, end_ds, universe, n_blocks).reset_index()
        for ds, data in factor_data.groupby('ds'):
            date = '-'.join([ds[:4], ds[4:6], ds[6:]])
            code_list = list(data.code.unique().to_array())
            factor_list = self._factors
            factor_path = f'/mnt/lustre/home/zwx/hft_signal_maker/factors/{self.name}'
            time_series = [("09:30:00", "11:30:00"), ("13:00:00", "15:00:00")]
            freq = _time_flag(self.time_flag)
            os.makedirs(factor_path, exist_ok=True)
            file_name = os.path.join(factor_path, f'{ds}.h5')
            writer = FactorH5Writer(file_name)
            writer.init(code_list=code_list,
                        factor_list=factor_list, time_series=time_series, date=date, freq=freq,
                        data_type="f")
            if freq < 60:
                time = [
                    int(f"{hour}{minute if minute >= 10 else f'0{minute}'}{second if second >= 10 else f'0{second}'}")
                    for hour in range(9, 16) for minute in range(60) for second in range(0, 60, freq)]
            else:
                time = [int(f"{hour}{minute if minute >= 10 else f'0{minute}'}00") for hour in range(9, 16) for minute
                        in range(0, 60, int(freq / 60))]
            time = [t for t in time if (93000 <= t <= 113000) | (130000 <= t <= 150000)]
            time = cudf.DataFrame({'time_flag': time, 'anchor': 1})
            codes = cudf.DataFrame({'code': code_list, 'anchor': 1})
            ct = codes.merge(time, on=['anchor'], how='outer').drop(columns=['anchor']).sort_values(
                ['code', 'time_flag']).reset_index(drop=True)
            data = ct.merge(data, on=['code', 'time_flag'], how='left').sort_values(['code', 'time_flag']).reset_index(
                drop=True)
            data[data.time_flag == 93000] = data[data.time_flag == 93000].fillna(0)
            data = data.to_pandas().fillna(method='ffill')
            data['time_flag'] = data.time_flag.astype('str')
            data['time_flag'] = data.time_flag.str[:-4] + ':' + data.time_flag.str[-4:-2] + ':' + data.time_flag.str[
                                                                                                  -2:]
            data_unstack = cudf.from_pandas(data.set_index(['code', 'time_flag'])).unstack()
            columns = [c for c in data.columns if c not in ('ds', 'code', 'time_flag')]
            tensor_result = []
            for c in columns:
                tensor_result.append(cp.asarray(data_unstack[c].as_gpu_matrix()))
            tensor_result = cp.stack(tensor_result)
            tensor_result = tensor_result.swapaxes(2, 0)
            writer.h5_file['data'][:, :, :] = np.array(tensor_result.astype('float32').get())
            writer.close()

    def compute(self, start_ds, end_ds, universe='StockA', n_blocks=1, window=1):
        """
        ???????????????????????????

        :param window:
        :param start_ds: ????????????
        :param end_ds: ????????????
        :param universe: ?????????
        :param n_blocks: ??????block??????????????????????????????????????????????????????????????????????????????????????????
        :return: ??????????????????????????????cudf.DataFrame????????????
        """
        import sys
        sys.path.append('/mnt/lustre/app')
        from dataapi.stock.cudf_market import get_cudf_transaction, get_cudf_snapshot, get_cudf_order_sh, \
            get_cudf_order_sz
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
                if universe in ('ALL', 'TOP2000', 'ALL_BASE_WITH_KECHUANG', 'ALL_BASE', 'QL1', 'SZ50', 'HS300', 'ZZ500', 'ZZ1000'):
                    import dataapi
                    code_list = list(dataapi.get_stock_universe(ds, ds, universe).reset_index()['code'].unique())
                elif universe in ('all', 'tiny', 'tinybuffer', 'tinyall', 'broad800', 'broad800buffer', 'broad800all', 'exipo60', 'StockA'):
                    from team_api.data.data_getter import api
                    code_list = list(api.universe(ds, universe).reset_index()['code'].unique())
                else:
                    raise NotImplementedError(f'unknown universe {universe}')
            if self.include_snap:
                logbook.info(f'start load snap of {ds} and cut to {n_blocks} blocks')
                snap = []
                for i in range(window):
                    temp_ds = str(int(ds) - i)
                    snap.append(get_cudf_snapshot(
                        temp_ds, code=code_list, source='huatai', ns_time=False))
                context._add_snapshot_blocks(_cdf_cut_to_host(cudf.concat(snap), n_blocks))
                logbook.info(f'finish load snap')
            if self.include_trans:
                logbook.info(f'start load trans of {ds} and cut to {n_blocks} blocks')
                context._add_trans_blocks(_cdf_cut_to_host(get_cudf_transaction(
                    ds, code=code_list, source='rough_merge_v0', ns_time=False), n_blocks))
                logbook.info(f'finish load trans')
            if self.include_order:
                logbook.info(f'start load order of {ds} and cut to {n_blocks} blocks')
                order_sh = get_cudf_order_sh(ds, code=code_list, source='kuanrui', ).drop(
                    columns=["orderindex", "channel", "bizindex"])
                order_sh['exchange'] = 'sh'
                order_sz = get_cudf_order_sz(ds, code=code_list, source='rough_merge_v0', )
                order_sz['exchange'] = 'sz'
                order = cudf.concat([order_sh, order_sz])
                context._add_order_blocks(_cdf_cut_to_host(order, n_blocks))
                logbook.info(f'finish load order')
            if self.include_trans_wiz_order:
                logbook.info(f'start load trans_wiz_order of {ds} and cut to {n_blocks} blocks')
                order_sz = get_cudf_order_sz(ds, source='huatai')
                order_sh = get_cudf_order_sh(ds, source='kuanrui')
                bid_sz = order_sz[order_sz.bsflag == 1][['time', 'code', 'orderid', 'price', 'volume']]
                ask_sz = order_sz[order_sz.bsflag == -1][['time', 'code', 'orderid', 'price', 'volume']]
                bid_sh = order_sh[order_sh.bsflag == 1][['time', 'code', 'orderid', 'price', 'volume']]
                ask_sh = order_sh[order_sh.bsflag == -1][['time', 'code', 'orderid', 'price', 'volume']]
                del order_sz, order_sh
                bid_columns = ['bid_time', 'code', 'bidID', 'bid_price', 'bid_volume']
                ask_columns = ['ask_time', 'code', 'askID', 'ask_price', 'ask_volume']
                bid_sz.columns = bid_columns
                bid_sh.columns = bid_columns
                ask_sz.columns = ask_columns
                ask_sh.columns = ask_columns
                trans = get_cudf_transaction('20210607', source='rough_merge_v0')
                bid = cudf.concat([bid_sz, bid_sh])
                del bid_sz, bid_sh
                ask = cudf.concat([ask_sz, ask_sh])
                del ask_sz, ask_sh
                trans = trans.merge(bid, how='left', on=['code', 'bidID'])
                del bid
                trans = trans.merge(ask, how='left', on=['code', 'askID'])
                del ask
                trans = trans.set_index(['ds', 'code', 'time']).sort_index()
                context._add_trans_wiz_order_blocks(_cdf_cut_to_host(trans, n_blocks))
                logbook.info(f'finish load trans_wiz_order')
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
                    res = func(context)
                    res = res.reset_index()
                    res['ds'] = ds
                    res.append(res.set_index(['ds', 'code', 'time_flag']))
                    context._frame_data = cudf.concat([context._frame_data, res], axis=1)
                else:
                    raise NotImplementedError()
            df_result.append(context._frame_data.reset_index().to_arrow())
            logbook.info(f'finish compute {ds}')
        return cudf.concat([cudf.DataFrame.from_arrow(df) for df in df_result]).set_index(
            ['ds', 'code', 'time_flag'])[self._factors]

    def add_block_step(self, func, **kwargs):
        """
        ??????block?????????block??????????????????????????????????????????????????????

        :param func: HftContext => cudf.DataFrame(index with (code, time_freq))???????????????block?????????????????????
        :param kwargs:
        :return:
        """
        self._steps.append(('block', func, kwargs))

    def add_cross_step(self, func, **kwargs):
        """
        ??????cross????????????cross????????????????????????????????????????????????

        :param func: HftContext => cudf.DataFrame(index with (code, time_freq))
        :param kwargs:
        :return:
        """
        self._steps.append(('cross', func, kwargs))

    def gen_factors(self, factors):
        """
        ?????????????????????????????????

        :param factors:
        :return:
        """
        self._factors = factors

    def get(self, start_ds, end_ds, time_range: list = None, code_list: list = None, factor_list: list = None, n_jobs=None):
        """
        ???????????????h5???????????????

        :param start_ds: ????????????
        :param end_ds: ????????????
        :param time_range: ?????????????????????['09:30:00', '10:00:00']
        :param code_list: ?????????????????????['000002', '601888']
        :param factor_list: ?????????????????????['open', 'close', 'high', 'low']
        :param n_jobs: ??????????????????None
        :return: df
        """
        from tqdm import tqdm
        trading_days = du.get_between_date(start_ds, end_ds)
        if n_jobs:
            from joblib import Parallel, delayed
            df_list = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
                delayed(_read_one_day)(self.name, ds, time_range, code_list, factor_list)
                for ds in tqdm(trading_days))
        else:
            df_list = []
            for ds in tqdm(trading_days):
                df_list.append(_read_one_day(self.name, ds, time_range, code_list, factor_list))
        df = pd.concat(df_list)
        return df


if __name__ == '__main__':
    pass
