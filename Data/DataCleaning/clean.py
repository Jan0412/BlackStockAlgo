# libs for data handling
import pandas as pd
import numpy as np

from BlackStock_Libs.Load_status import ProcessStatus
import datetime

class Clean(object):
    __base_data:pd.DataFrame = None
    __NaN_idx:np.ndarray = None

    __ps:ProcessStatus = None

    def __init__(self, base_data:pd.DataFrame, **kwargs):
        self.__check_by_col:str = kwargs['by_col'] if 'by_col' in kwargs.keys() else base_data.columns.values[-1]
        self.__base_data = self.set_dateIndex(df=base_data)
        self.__NaN_idx = self.__base_data.index.values[self.get_NaN_values(values=self.__base_data[self.__check_by_col])]

        self.__ps = ProcessStatus(max_value=len(self.__NaN_idx), name='CleanByRef')

        print(f'Rows to replace {len(self.__NaN_idx)}')

    @staticmethod
    def set_dateIndex(df:pd.DataFrame):
        if not isinstance(df.index, pd.DatetimeIndex):
            df['Date'] = pd.to_datetime(arg=df['Date'])
            df = df.set_index(keys='Date', drop=True)
        return df

    @staticmethod
    def get_NaN_values(values):
        NaN_values = np.where(np.isnan(values))
        NaN_values = np.asarray(NaN_values)
        return NaN_values.flatten()

    def clean_by_ref(self, ref_data, **kwargs):
        ref_columns:list = []
        replaced_count:int = 0
        fail_count:int = 0

        if 'ref_columns' in kwargs.keys():
            for col in kwargs.get('ref_columns'):
                if any([col in df.columns.values for df in ref_data]) and col in self.__base_data.columns.values:
                    ref_columns.append(col)
            assert len(ref_columns) > 0, ""
        else:
            ref_columns = self.__base_data.columns.values

        ref_data = [self.set_dateIndex(df=ref_df) for ref_df in ref_data]
        ref_shape = (len(ref_data) - 1)

        for NaN_idx in self.__NaN_idx:
            for (c, ref_df) in enumerate(ref_data):
                try:
                    row = ref_df[ref_columns].loc[NaN_idx]
                    if not np.isnan(row.values).any():
                        for col in ref_columns:
                            self.__base_data.at[NaN_idx, col] = row[col]

                        replaced_count += 1
                        break

                    elif c >= ref_shape:
                        fail_count += 1

                except Exception as e:
                    print(f' RefDataset Num. {c} --> {e}')

            self.__ps.update()

        self.__ps.finish()
        print(f'Replaced values: {replaced_count} | Failed values: {fail_count}')

        return self.__base_data

    def clean_by_mean(self, columns:list):
        NaN_id = self.get_NaN_values(values=self.__base_data[self.__check_by_col])
        NaNs_in_row = find_rows(values=NaN_id, row_range=2)

        for idx in NaN_id[np.invert(NaNs_in_row)]:
            for column in columns:
                mean_value = (self.__base_data[column].iloc[(idx-1)] + self.__base_data[column].iloc[(idx+1)]) / 2
                self.__base_data[column].iloc[idx] = mean_value

        return self.__base_data


class CleanTimeline(object):
    df:pd.DataFrame = None
    start_date = None
    last_date = None
    freq_sek:int = None

    __ps:ProcessStatus = None

    def __init__(self, df:pd.DataFrame, freq_sek, **kwargs):
        self.freq_sek = freq_sek

        date_col:str = kwargs['date_col'] if 'date_col' in kwargs.keys() else 'Date'

        if not isinstance(df.index, pd.DatetimeIndex):
            df[date_col] = pd.to_datetime(arg=df[date_col])
            self.df = df.set_index(keys='Date', drop=True)

        self.start_date = self.df.index.values[0]
        self.last_date = self.df.index.values[-1]

        true_range = len(self.df.index.values)
        expected_range_ = self.expected_range(last_date=self.start_date, first_date= self.last_date, freq_sek=self.freq_sek)

        #self.__ps = ProcessStatus(max_value=(true_range - expected_range_), name='CleanTimeline')


    def __call__(self, set_timestamp:bool, *args, **kwargs):
        idx = 0
        current_date = self.start_date

        while current_date != self.last_date:
            try:
                if self.df.index.values[idx] != current_date:
                    new_value = [self.timestamp(cd=current_date)] if set_timestamp else []
                    new_value += [np.nan] * (len(self.df.columns.values) - 1 if set_timestamp else 0)

                    self.df = self.insert_row(df=self.df, row_idx=current_date, value=new_value)

            except Exception as e:
                print(e)

            current_date += np.timedelta64(self.freq_sek,'s')
            idx += 1
            print(current_date)
            #self.__ps.update()

        return self.df

    @staticmethod
    def insert_row(df:pd.DataFrame, row_idx, value):
        pre_df = df.loc[:row_idx]
        aft_df = df.loc[row_idx:]

        pre_idx = df.index.values[:pre_df.shape[0]]
        aft_idx = df.index.values[-aft_df.shape[0]:]

        add_df = pd.DataFrame({ col : [val] for val, col in zip(value, df.columns.values)})
        pre_df = pre_df.append(add_df)
        df_result = pd.concat([pre_df, aft_df])
        df_result.index = np.concatenate([pre_idx, [row_idx], aft_idx])

        return df_result

    @staticmethod
    def timestamp(cd):
        return int((cd - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))

    @staticmethod
    def expected_range(last_date, first_date, freq_sek):
        return int((last_date - first_date) / 1e9 / freq_sek)


def clear_timeline(df:pd.DataFrame, freq_sek:int, set_timestamp:bool, **kwargs):
    timestamp = lambda cd : int((cd - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
    true_range = lambda last_date, first_date : int((last_date - first_date) / 1e9 / freq_sek)

    def check_range(date_range:list):
        first_date = date_range[0]
        last_date = date_range[-1]

        expect_range = len(date_range)
        return expect_range == true_range(last_date=last_date, first_date=first_date)

    def insert_row(df, row_idx, value):
        pre_df = df.loc[:row_idx]
        aft_df = df.loc[row_idx:]

        pre_idx = df.index.values[:pre_df.shape[0]]
        aft_idx = df.index.values[-aft_df.shape[0]:]

        add_df = pd.DataFrame({ col : [val] for val, col in zip(value, df.columns.values)})
        pre_df = pre_df.append(add_df)
        df_result = pd.concat([pre_df, aft_df])
        df_result.index = np.concatenate([pre_idx, [row_idx], aft_idx])

        return df_result

    date_col:str = kwargs['date_col'] if 'date_col' in kwargs.keys() else 'Date'

    if not isinstance(df.index, pd.DatetimeIndex):
        df[date_col] = pd.to_datetime(arg=df[date_col])
        df = df.set_index(keys='Date', drop=True)

    if not check_range(date_range=df.index.values):
        current_date = df.index.values[0]
        last_idx = df.index.values[-1]

        idx = 0
        while current_date != last_idx:
            try:
                if df.index.values[idx] != current_date:
                    new_value = [timestamp(cd=current_date)] if set_timestamp else []
                    new_value += [np.nan] * (len(df.columns.values) - 1 if set_timestamp else 0)

                    df = insert_row(df=df, row_idx=current_date, value=new_value)
            except Exception as e:
                print(e)

            current_date += np.timedelta64(freq_sek,'s')
            idx += 1
    check_range(date_range=df.index.values)
    return df


def find_rows(values:np.ndarray, row_range):
    values_in_row = []

    for idx in range(0, values.shape[0]):
        left_bound = (idx - row_range // 2)
        right_bound = (idx + row_range // 2)

        if left_bound < 0:
            right_bound = (right_bound + abs(left_bound))
            left_bound = 0

        elif right_bound > values.shape[0]:
            left_bound = (idx + abs(left_bound - values.shape[0]))
            right_bound = values.shape[0]

        if (np.mean(np.diff(values[left_bound:right_bound]))) == 1:
            if (row_range // 2) > 1:
                for i in range(1, (row_range // 2)):
                    values_in_row[-i] = True
            else:
                values_in_row.append(True)
                try:
                    values_in_row[-2] = True
                except Exception as e:
                    print(e)
                    pass
        else:
            values_in_row.append(False)

    return values_in_row
