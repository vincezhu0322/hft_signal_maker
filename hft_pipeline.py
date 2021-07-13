from team_api import date_util as du
from team_api.data.data_getter import api
from dataapi.stock.cudf_market import get_cudf_transaction, get_cudf_snapshot, get_cudf_order_sh, get_cudf_order_sz
import cudf
import logbook
import pandas as pd


pd.options.display.max_columns = 20


class HftContext:

    def __init__(self):
        self.ds = None
        self.trans_data = None
        self.order_data = None
        self.snap_data = None
        self.result_data = []

    def get_trans(self, dimension_fix=True, with_minute_flag=False, exclude_auction=False, exclude_cancel=True):
        if dimension_fix:
            self.trans_data['price'] = self.trans_data['price'] / 10000
        if with_minute_flag:
            self.trans_data['minute_flag'] = self.trans_data['time'].map(lambda x: (x - 1) // 100000 + 1)
        if exclude_auction:
            self.trans_data = self.trans_data[self.trans_data['time'] >= 93000000]
        if exclude_cancel:
            self.trans_data = self.trans_data[self.trans_data['transType'] != 0]
        return self.trans_data

    def get_snap(self):
        pass

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

    def compute(self, start_ds, end_ds, universe='StockA', n_code_partition=1):
        assert n_code_partition == 1, NotImplementedError()
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
                snap = get_cudf_snapshot(ds, code=code_list, source='rough_merge')
                context.snap_data = snap
            if self.include_trans:
                trans = get_cudf_transaction(ds, code=code_list, source='rough_merge_v0')
                context.trans_data = trans
            if self.include_order:
                raise NotImplementedError()
            for step in self._steps:
                func_type, func, kwargs = step
                if func_type == 'block':
                    res = func(context)
                    res = res.reset_index()
                    res['ds'] = ds
                    context.result_data.append(res.set_index(['ds', 'code', 'minute_flag']))
                else:
                    raise NotImplementedError()
            ds_res = cudf.concat(context.result_data).sort_index().to_pandas()
            result.append(ds_res)
            logbook.info(f'finish compute {ds}')
        return pd.concat(result)

    def add_block_step(self, func, **kwargs):
        self._steps.append(('block', func, kwargs))


if __name__ == '__main__':
    pass


