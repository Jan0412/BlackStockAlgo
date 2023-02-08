# Imports
import numpy as np
import sqlite3 as sql
import math
import matplotlib.pyplot as plt


def savitzky_golay_smoothing(data, window_size: int, order: int, derivative: int) -> np.ndarray:
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError('window size must be a positive odd number')

    if window_size < (order + 2):
        raise TypeError('window size is too small for the polynomial')

    half_window = (window_size - 1) // 2

    b = np.mat([[k ** i for i in range(order + 1)] for k in range(-half_window, (half_window + 1))])
    m = np.linalg.pinv(b).A[derivative] * math.factorial(derivative)

    firstvals = data[0] - np.abs(data[1:half_window + 1][::-1] - data[0])
    lastvals = data[-1] + np.abs(data[-half_window - 1:-1][::-1] - data[-1])
    data = np.concatenate((firstvals, data, lastvals))

    return np.convolve(m[::-1], data, mode='valid')


def get_extrema(data, window_range: int, window_size: int = 21, order: int = 5) -> (np.ndarray, np.ndarray):
    first_deriv = savitzky_golay_smoothing(data=data, window_size=window_size, order=order, derivative=1)
    second_deriv = savitzky_golay_smoothing(data=data, window_size=window_size, order=order, derivative=2)

    extrema = find_zero_cut(data=first_deriv)
    maximum = np.logical_and(extrema, second_deriv < 0)
    minimum = np.logical_and(extrema, second_deriv > 0)
    maximum = np.argwhere(maximum == True)[:, 0]
    minimum = np.argwhere(minimum == True)[:, 0]

    local_max_id = [np.argmax(data[(idx - window_range):(idx + window_range + 1)]) + (idx - window_range)
                    for idx in maximum if (idx > window_range) and (idx < (len(data) - window_range + 1))]
    local_min_id = [np.argmin(data[(idx - window_range):(idx + window_range + 1)]) + (idx - window_range)
                    for idx in minimum if (idx > window_range) and (idx < (len(data) - window_range + 1))]

    return np.asarray(np.unique(local_max_id)), np.asarray(np.unique(local_min_id))


def find_zero_cut(data: np.ndarray, axis: int = 0) -> np.ndarray:
    locs = np.arange(0, data.shape[axis])
    plus = data.take(indices=(locs + 1), mode='clip')

    lh = np.ones(shape=data.shape, dtype=bool)
    hl = np.ones(shape=data.shape, dtype=bool)

    lh &= data < 0
    lh &= 0 < plus

    hl &= data > 0
    hl &= 0 > plus

    return np.logical_or(lh, hl)


def min_period(period: int, max_ids: np.ndarray, min_ids: np.ndarray) -> (np.ndarray, np.ndarray):
    tmp = np.concatenate((max_ids, min_ids))
    tmp_index = np.argsort(tmp)
    f = [abs(tmp[tmp_index[i - 1]] - tmp[tmp_index[i]]) > period for i in range(1, tmp_index.shape[0])]
    f.insert(0, True)

    tmp_clean = tmp_index[f]
    split_index = max_ids.shape[0]
    max_ids = tmp[tmp_clean[np.where(tmp_clean < split_index)]]
    min_ids = tmp[tmp_clean[np.where(tmp_clean >= split_index)]]

    return max_ids, min_ids


PATH_BTC_DB = "../Data/Bitcoin.db"
WINDOW_RANGE = 100

def test() -> None:
    connection = sql.connect(database=PATH_BTC_DB)
    connection.row_factory = lambda cur, row: row[0]
    cursor = connection.cursor()
    close = cursor.execute('SELECT Close FROM Bitcoin').fetchall()
    close = np.asarray(close[:16_596])
    connection.close()

    max_id, min_id = get_extrema(data=close, window_range=WINDOW_RANGE)
    max_id, min_id = min_period(period=5, max_ids=max_id, min_ids=min_id)
    time = np.arange(0, len(close))

    plt.plot(time, close, color='blue', label='Close Price')
    plt.scatter(time[max_id], close[max_id], color='red', marker='v', label='Buy')
    plt.scatter(time[min_id], close[min_id], color='green', marker='^', label='Sell')

    plt.show()


if __name__ == '__main__':
    test()
