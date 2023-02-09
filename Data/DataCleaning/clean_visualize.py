# libs for data visualizing
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick2_ohlc
import seaborn as sns

# libs for data handling
import numpy as np
import pandas as pd
import sqlite3 as sql
from DataCleaning.clean import find_rows


def plot_NaN(values:np.ndarray, ax, **kargs):
    color:str = kargs['col'] if 'col' in kargs.keys() else 'r'
    linewidth:int =  kargs['width'] if 'width' in kargs.keys() else 1
    NaN_values = np.where(np.isnan(values))
    NaN_values = np.asarray(NaN_values)
    NaN_values = NaN_values.flatten()

    values_in_row = find_rows(values=NaN_values, row_range=2)

    for NaN, row_check in zip(NaN_values, values_in_row):
        if row_check:
            ax.axvline(x=NaN, linewidth=linewidth, color='r')
        else:
            ax.axvline(x=NaN, linewidth=linewidth, color='g')


def plot_candlestick(ax, opens, highs, lows, closes, **kargs):
    width:int = kargs['width'] if 'width' in kargs.keys() else 2
    color_up:str = kargs['up'] if 'up' in kargs.keys() else 'g'
    color_down:str = kargs['down'] if 'down' in kargs.keys() else 'r'
    alpha:float = kargs['alpha'] if 'alpha' in kargs.keys() else 1

    candlestick2_ohlc(ax=ax, opens=opens, highs=highs, lows=lows, closes=closes,
                      width=width, colorup=color_up, colordown=color_down, alpha=alpha)


def test():
    con = sql.connect(database='..\Data\Bitcoin.db')
    df = pd.read_sql(sql='SELECT * FROM Bitcoin', con=con)
    _, ax = plt.subplots(nrows=1, ncols=1)
    plot_NaN(values=df['High'], ax=ax)
    plot_candlestick(ax=ax, opens=df['Open'], highs=df['High'], lows=df['Low'], closes=df['Close'])

    plt.show()


if '__main__' == __name__:
    test()
