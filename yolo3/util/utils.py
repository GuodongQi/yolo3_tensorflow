import numpy as np


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sec2time(sec, n_msec=3):
    ''' Convert seconds to 'D days, HH:MM:SS.FFF' '''
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if n_msec > 0:
        pattern = '%%02dh %%02dm %%0%d.%dfs' % (n_msec + 3, n_msec)
    else:
        pattern = r'%02dh %02dm %02s'
    if d == 0:
        return pattern % (h, m, s)
    return ('%d d, ' + pattern) % (d, h, m, s)
