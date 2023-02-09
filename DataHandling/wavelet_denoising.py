# libs for data handling
import pandas as pd
import numpy as np

import sqlite3 as sql

import pywt

# libs for data visualising
import matplotlib.pyplot as plt

def discrete_wavelet_transform(data:list, wavelet:str,  coeff_level:int=2, mode_th:str='hard', mode_wav:str='periodization'):
    """
    :param data:
    :param wavelet:
    :param coeff_level:
    :param mode_th:
    :param mode_wav:
    :return:
    """
    data = np.asarray(data)

    madev = lambda d : np.mean(np.absolute(d - np.mean(d)))

    max_level = pywt.dwt_max_level(data_len=data.shape[0], filter_len=wavelet)
    coeff = pywt.wavedec(data=data, wavelet=wavelet, level=max_level, mode=mode_wav)
    sigma = (1 / 0.6745) * madev(coeff[-coeff_level])
    thresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = (pywt.threshold(data=c, value=thresh, mode=mode_th) for c in coeff[1:])

    return pywt.waverec(coeffs=coeff, wavelet=wavelet, mode=mode_wav)

def continuous_wavelet_transform(data, wavelet):
    """
    :param data:
    :param wavelet:
    :return:
    """
    scales = pywt.scale2frequency(wavelet=wavelet, scale=np.arange(1, 300)) / 1e-3
    coefs, freq = pywt.cwt(data=data, scales=scales, wavelet=wavelet)
    return np.abs(coefs) ** 2


PATH_BTC_DB = '../Data/Bitcoin.db'

def test():
    connection = sql.connect(database=PATH_BTC_DB)
    df = pd.read_sql(sql='SELECT Close FROM Bitcoin', con=connection)
    df = df['Close'].values

    original_ary = np.asarray(df)
    time = np.arange(start=0, stop=original_ary.shape[0])

    print(original_ary.shape, time.shape)


    x_windows = 6
    y_windows = 4
    windows_len = int(df.shape[0] / (x_windows * y_windows / 2))
    fig, ax = plt.subplots(nrows=y_windows, ncols=x_windows)

    c = 0
    for i in range(0, y_windows, 2):
        for j in range(x_windows):
            start = c * int(windows_len / 1)
            stop = start + windows_len
            print(f'[i, j]: {i, j} -> {start, stop}')
            ary = original_ary[start:start+windows_len]
            noise_free_data = discrete_wavelet_transform(data=ary, wavelet='db10', coeff_level=1)
            log_return = np.diff(np.log(noise_free_data))
            log_return = (log_return - np.mean(log_return)) / np.std(log_return)
            power = continuous_wavelet_transform(data=log_return, wavelet='cmor1.1-2.0')
            norm_power = (power - np.min(power)) / (np.max(power) - np.min(power))
            ax[i, j].plot(log_return)
            ax[i+1, j].imshow(norm_power, aspect='auto')

            c += 1

    plt.show()

if __name__ == '__main__':
    test()
