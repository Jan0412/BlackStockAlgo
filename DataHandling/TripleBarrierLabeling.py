import numpy as np
import pandas as pd
import sqlite3 as sql
from itertools import groupby
from scipy import stats
import time
import matplotlib.pyplot as plt


def calcVol(close, delta, span=100) -> np.ndarray:
    df = close.index.searchsorted(close.index - delta)
    df = df[df > 0]
    df = pd.Series(close.index[df - 1], index=close.index[close.shape[0] - df.shape[0]:])
    df = close.loc[df.index] / close.loc[df.values].values - 1
    df = df.ewm(span=span).std()
    return np.asarray(df.values)


def calcATR(OHLC: pd.DataFrame, win) -> np.ndarray:
    art_ary = np.empty(shape=OHLC.shape[0])
    high = pd.Series(OHLC['High'].rolling(win, min_periods=win))
    low = pd.Series(OHLC['Low'].rolling(win, min_periods=win))
    close = pd.Series(OHLC['Close'].rolling(win, min_periods=win))

    for i in range(len(OHLC.index)):
        tr = np.max([(high[i] - low[i]), np.abs(high[i] - close[i]), np.abs(low[i] - close[i])], axis=0)
        art_ary[i] = tr.sum() / win

    return art_ary


def tripleBarrierLabeling(data: pd.DataFrame, vb, ub: float, lb: float) -> pd.DataFrame:
    """
    :param ub: upper boundary thresholds
    :param lb: lower boundary thresholds
    :param data: Dataframe must contain the column Close
    :param vb: set the vertical barrier
    :return: Labels 1 -> Long, 0 -> Neutral, -1 -> Short
    """

    t2 = data.index.searchsorted(data.index + vb)
    t2 = data.index.values[t2[t2 < data.shape[0]]]
    t1 = data.index.values[:t2.shape[0]]

    def neutral(i: int) -> None:
        data.at[i, 'label'] = 0

    for a, b in zip(t1, t2):
        sequence = data.Close.loc[a:b].values
        med = np.median(sequence)
        std = np.std(sequence)
        vol = data.vol.loc[a]

        if (med + std) > sequence[0]:
            upper_boundary = sequence[0] + sequence[0] * ub * vol
            if (sequence >= upper_boundary).any():
                data.at[a, 'label'] = 1
            else:
                neutral(i=a)

        elif (med - std) < sequence[0]:
            lower_boundary = sequence[0] - sequence[0] * lb * vol
            if (sequence <= lower_boundary).any():
                data.at[a, 'label'] = -1
            else:
                neutral(i=a)
        else:
            neutral(i=a)

    return data


def smooth_labels(data: pd.DataFrame, gs=1, w=1) -> pd.DataFrame:
    for label in range(-1, 1):
        id_val = data[data.label == label].index.values
        for _, g in groupby(enumerate(id_val), lambda ix: (ix[0] - ix[1])):
            group = np.asarray(list(g))[:, 1]
            if group.shape[0] <= gs:
                seq = data.label.iloc[(group[0] - w):(group[-1] + w + 1)].values
                if all(val in seq for val in [-1, 0, 1]):
                    if seq[w] == -1 or seq[w] == 1:
                        mode = 0
                    else:
                        mode = seq[0]
                else:
                    mode = stats.mode(seq)[0]
                data.at[group[0], 'label'] = mode

    return data


PATH_BTC_DB = '../Data/Bitcoin.db'
VB_OFFSET = 8
UB_THRESHOLD = 0.6
LB_THRESHOLD = 0.5
DELTA = 13

def test() -> None:
    t0 = time.time()
    connection = sql.connect(database=PATH_BTC_DB)
    df = pd.read_sql(sql='SELECT Timestamp, Open, High, Low, Close FROM Bitcoin', con=connection)
    df = df.iloc[:10_00]

    vol = [np.nan] * (DELTA + 1)
    vol += list(calcVol(close=df.Close, delta=DELTA))
    df = df.assign(vol=vol)
    df = tripleBarrierLabeling(data=df, vb=VB_OFFSET, ub=UB_THRESHOLD, lb=LB_THRESHOLD)

    #df = smooth_labels(data=df)
    t1 = time.time()
    print(f'Time to label: {t1 - t0}s')

    print(f"Nan labels count: {np.count_nonzero(np.isnan(df.label.values))}")
    print(df)

    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=1, ncols=1)
    for (l, c) in [(-1, 'red'), (0, 'yellow'), (1, 'green')]:
        axes.scatter(df[df.label == l].index.values, df[df.label == l].Close.values, linewidths=0.5, color=c)
    axes.plot(df['Close'].values, color='blue')
    plt.show()


if __name__ == '__main__':
    test()
