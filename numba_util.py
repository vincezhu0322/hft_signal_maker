import numba


@numba.jit(nopython=True)
def _ts_align(ts: int, freq_second: int):
    second = ts % 100
    minute = (ts // 100) % 100
    hour = ts // 10000
    if second > 60 - freq_second:
        minute, second = minute + 1, 0
    else:
        second = ((second - 1) // freq_second + 1) * freq_second
    if minute >= 60:
        hour, minute = hour + 1, 0
    return hour * 10000 + minute * 100 + second


def get_cudf_from_arrow(arrow_data, align_step):
    import cudf
    res = cudf.DataFrame.from_arrow(arrow_data)
    res['time_flag'] = res['time'].map(lambda x: _ts_align(x // 1000, align_step))
    return res
