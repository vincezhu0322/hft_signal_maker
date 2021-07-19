import numba


from numba.pycc import CC
export_helper = CC("numba_tool")


@export_helper.export("numba_ts_align", "i4(i4, i4)")
@numba.njit()
def numba_ts_align(ts: int, freq_second: int):
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


if __name__ == '__main__':
    export_helper.compile()