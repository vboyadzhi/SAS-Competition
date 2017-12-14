# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import timeit
from itertools import product


def to_time(df):
    df['DTIME_CREDIT'] = pd.to_datetime(df['DTIME_CREDIT'], format='%d.%m.%Y')
    df['DTIME_CREDIT_ENDDATE_FACT'] = pd.to_datetime(df['DTIME_CREDIT_ENDDATE_FACT'], format='%d.%m.%Y')
    df['DTIME_CREDIT_ENDDATE'] = pd.to_datetime(df['DTIME_CREDIT_ENDDATE'], format='%d.%m.%Y')
    df['DTIME_CREDIT_UPDATE'] = pd.to_datetime(df['DTIME_CREDIT_UPDATE'], format='%d.%m.%Y')
    df['SK_DATE_DECISION'] = pd.to_datetime(df['SK_DATE_DECISION'], format='%Y%m%d')
    return df


def drop_duplicates(df):
    mask = df[['ID', 'NUM_SOURCE', 'CREDIT_CURRENCY', 'DTIME_CREDIT', 'AMT_CREDIT_SUM', 'CREDIT_TYPE']]\
        .sort_values(['ID', 'DTIME_CREDIT', 'NUM_SOURCE'], ascending=[True, False, False])\
        .drop('NUM_SOURCE', axis=1)\
        .duplicated()
    df = df.loc[~mask]
    return df


def clean_trash(df):
    # несколько случаев когда берет подряд 2 кредита и статус не успел проставится - ставлю Х
    df['TEXT_PAYMENT_DISCIPLINE'] = df['TEXT_PAYMENT_DISCIPLINE'].fillna('X')
    # какой-то трешак в данных
    df['TEXT_PAYMENT_DISCIPLINE'] = df['TEXT_PAYMENT_DISCIPLINE'].map(lambda x: re.sub('\..+', '', x))
    return df


# считаю количество платежей
def count_payments(df):
    df['zeros'] = df['TEXT_PAYMENT_DISCIPLINE'].str.replace('C', 'q').str.replace('X', 'q').str.replace('1', 'q')\
        .str.replace('2', 'q').str.replace('3', 'q').str.replace('4', 'q').str.replace('5', 'q')
    df['ones'] = df['TEXT_PAYMENT_DISCIPLINE'].str.replace('C', 'q').str.replace('X', 'q').str.replace('0', 'q')\
        .str.replace('2', 'q').str.replace('3', 'q').str.replace('4', 'q').str.replace('5', 'q')
    df['twos'] = df['TEXT_PAYMENT_DISCIPLINE'].str.replace('C', 'q').str.replace('X', 'q').str.replace('1', 'q')\
        .str.replace('0', 'q').str.replace('3', 'q').str.replace('4', 'q').str.replace('5', 'q')
    df['threes'] = df['TEXT_PAYMENT_DISCIPLINE'].str.replace('C', 'q').str.replace('X', 'q').str.replace('1', 'q')\
        .str.replace('2', 'q').str.replace('0', 'q').str.replace('4', 'q').str.replace('5', 'q')
    df['fours'] = df['TEXT_PAYMENT_DISCIPLINE'].str.replace('C', 'q').str.replace('X', 'q').str.replace('1', 'q')\
        .str.replace('2', 'q').str.replace('3', 'q').str.replace('0', 'q').str.replace('5', 'q')
    df['fives'] = df['TEXT_PAYMENT_DISCIPLINE'].str.replace('C', 'q').str.replace('X', 'q').str.replace('1', 'q')\
        .str.replace('2', 'q').str.replace('3', 'q').str.replace('4', 'q').str.replace('0', 'q')
    df['bads'] = df['TEXT_PAYMENT_DISCIPLINE'].str.replace('C', 'q').str.replace('X', 'q').str.replace('0', 'q')
    df['base'] = df['TEXT_PAYMENT_DISCIPLINE'].str.replace('C', 'q').str.replace('X', 'q')
    return df


# худший статус за периоды
def worst_status(df):
    df['worst_status_3m'] = df['TEXT_PAYMENT_DISCIPLINE'].str[:3].str.replace('C', '0').str.replace('X', '0').map(
        lambda x: int(max(str(x))))
    df['worst_status_6m'] = df['TEXT_PAYMENT_DISCIPLINE'].str[:6].str.replace('C', '0').str.replace('X', '0').map(
        lambda x: int(max(str(x))))
    df['worst_status_12m'] = df['TEXT_PAYMENT_DISCIPLINE'].str[:12].str.replace('C', '0').str.replace('X', '0').map(
        lambda x: int(max(str(x))))
    df['worst_status_24m'] = df['TEXT_PAYMENT_DISCIPLINE'].str[:24].str.replace('C', '0').str.replace('X', '0').map(
        lambda x: int(max(str(x))))
    df['worst_status_36m'] = df['TEXT_PAYMENT_DISCIPLINE'].str[:36].str.replace('C', '0').str.replace('X', '0').map(
        lambda x: int(max(str(x))))
    df['worst_status_48m'] = df['TEXT_PAYMENT_DISCIPLINE'].str[:48].str.replace('C', '0').str.replace('X', '0').map(
        lambda x: int(max(str(x))))
    df['worst_status_60m'] = df['TEXT_PAYMENT_DISCIPLINE'].str[:60].str.replace('C', '0').str.replace('X', '0').map(
        lambda x: int(max(str(x))))
    return df


# считаю количество разных платежей за периоды
def features_1(df):
    cols = []
    periods = [0, 0, 0, 0, 0, 0, 0]  # [0,3,6,12,24,36,48]
    periods2 = [3, 6, 12, 24, 36, 48, 60]
    columns = ['zeros', 'ones', 'twos', 'threes', 'fours', 'fives', 'bads', 'base']
    for col, (per, per2) in product(columns, zip(periods, periods2)):
        df[col + '_' + str(per) + ':' + str(per2)] = df[col].str[per:per2].str.replace('q', '').map(
            lambda x: len(x))
        cols.append(col + '_' + str(per) + ':' + str(per2))
    for col in columns:
        df[col] = df[col].str.replace('q', '').map(lambda x: len(x))
    return df, cols


def time_features(df):
    df['plan_lifetime'] = (df['DTIME_CREDIT_ENDDATE'] - df['DTIME_CREDIT']).map(
        lambda x: x.days if x == x else np.nan) / 365
    df['fact_lifetime'] = (df['DTIME_CREDIT_ENDDATE_FACT'] - df['DTIME_CREDIT']).map(
        lambda x: x.days if x == x else np.nan) / 365
    df['preschedule_time'] = (df['DTIME_CREDIT_ENDDATE'] - df['DTIME_CREDIT_ENDDATE_FACT']).map(
        lambda x: x.days if x == x else np.nan) / 365
    df['lastupdate_time'] = (df['SK_DATE_DECISION'] - df['DTIME_CREDIT_UPDATE']).map(
        lambda x: x.days if x == x else np.nan) / 365
    df['lastcredit_time'] = (df['SK_DATE_DECISION'] - df['DTIME_CREDIT']).map(
        lambda x: x.days if x == x else np.nan) / 365
    df['lastclosed_time'] = (df['SK_DATE_DECISION'] - df['DTIME_CREDIT_ENDDATE_FACT']).map(
        lambda x: x.days if x == x else np.nan) / 365
    return df


def agg_simple(df, cols):
    sum_cols, sum_cols2, max_cols, mean_cols, min_cols = cols
    x = df.groupby(['ID'])[sum_cols].sum().reset_index(drop=True)
    x.columns = x.columns.map(lambda y: 'sum' + '_' + str(y))
    x2 = df.groupby(['ID'])[sum_cols2].sum().reset_index(drop=True)
    x2.columns = x2.columns.map(lambda y: 'sum' + '_' + str(y))
    x3 = df.groupby(['ID'])[max_cols].max().reset_index(drop=True)
    x3.columns = x3.columns.map(lambda y: 'max' + '_' + str(y))
    x4 = df.groupby(['ID'])[mean_cols].mean().reset_index(drop=True)
    x4.columns = x4.columns.map(lambda y: 'mean' + '_' + str(y))
    x5 = df.groupby(['ID'])[min_cols].mean().reset_index(drop=True)
    x5.columns = x5.columns.map(lambda y: 'min' + '_' + str(y))
    x6 = df.groupby(['ID'])['SK_DATE_DECISION'].count().reset_index(drop=True)
    x6.name = 'count'
    return pd.concat([x, x2, x3, x4, x5, x6], axis=1)


def cross_features(df, df_source):
    def groupby_col(df_func, col_groupby, col_value, agg):
        func = {'sum': sum, 'count': np.count_nonzero, 'min': min, 'max': max, 'mean': np.mean}[agg]
        temp = df_func[['ID', col_groupby, col_value]].groupby(['ID', col_groupby]).agg(func) \
            .reset_index().pivot(index='ID', columns=col_groupby, values=col_value)
        temp.columns = temp.columns.map(lambda x: agg + '_' + col_groupby + '_' + col_value + '_' + str(x))
        return temp.reset_index(drop=True)

    # кросс min
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_ACTIVE', 'lastupdate_time', 'min')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_ACTIVE', 'lastcredit_time', 'min')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_CURRENCY', 'lastupdate_time', 'min')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_CURRENCY', 'lastcredit_time', 'min')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_TYPE', 'lastupdate_time', 'min')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_TYPE', 'lastcredit_time', 'min')], axis=1)
    # кросс max
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_MAX_OVERDUE', 'max')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_MAX_OVERDUE', 'max')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_MAX_OVERDUE', 'max')], axis=1)
    # кросс count
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_CURRENCY', 'SK_DATE_DECISION', 'count')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_ACTIVE', 'SK_DATE_DECISION', 'count')], axis=1)
    # кросс sum
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM', 'sum')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_ANNUITY', 'sum')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_SUM', 'sum')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_SUM_DEBT', 'sum')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_SUM', 'sum')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_TYPE', 'CREDIT_COLLATERAL', 'sum')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_SUM_DEBT', 'sum')], axis=1)
    # кросс mean
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_ACTIVE', 'preschedule_time', 'mean')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_CURRENCY', 'preschedule_time', 'mean')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_TYPE', 'preschedule_time', 'mean')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_ACTIVE', 'plan_lifetime', 'mean')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_CURRENCY', 'plan_lifetime', 'mean')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_TYPE', 'plan_lifetime', 'mean')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_ACTIVE', 'fact_lifetime', 'mean')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_CURRENCY', 'fact_lifetime', 'mean')], axis=1)
    df = pd.concat([df, groupby_col(df_source, 'CREDIT_TYPE', 'fact_lifetime', 'mean')], axis=1)
    return df


def double_cross_features(df, df_source):
    def groupby_col(df_func, col_groupby, col_value, agg):
        func = {'sum': sum, 'count': np.count_nonzero, 'min': min, 'max': max, 'mean': np.mean}[agg]
        col_groupby1, col_groupby2 = col_groupby
        temp = df_func[['ID', col_value, col_groupby1, col_groupby2]].groupby(
            ['ID', col_groupby1, col_groupby2]).agg(func).reset_index()
        temp['cols'] = temp[col_groupby1].astype(str) + '_' + temp[col_groupby2].astype(str)
        temp = temp.pivot(index='ID', columns='cols', values=col_value)
        temp.columns = temp.columns.map(
            lambda x: agg + '_' + col_groupby1 + '_' + col_groupby2 + '_' + col_value + '_' + str(x))
        return temp.reset_index(drop=True)

    df = pd.concat([df, groupby_col(df_source, ['CREDIT_ACTIVE', 'CREDIT_TYPE'], 'AMT_CREDIT_SUM', 'mean')], axis=1)
    return df


if __name__ == "__main__":
    RANDOM_STATE = 2017

    start_time1 = timeit.default_timer()
    start_time = timeit.default_timer()
    test = pd.read_csv('data/test.csv', sep=',')
    train = pd.read_csv('data/train.csv', sep=',')
    print('read data', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()

    train = to_time(train)
    test = to_time(test)
    print('to_time', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()

    train = drop_duplicates(train)
    test = drop_duplicates(test)
    print('drop_duplicates', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()

    train = clean_trash(train)
    test = clean_trash(test)
    print('clean_trash', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()

    train = count_payments(train)
    test = count_payments(test)
    print('count_payments', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()

    train = worst_status(train)
    test = worst_status(test)
    print('worst_status', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()

    train = time_features(train)
    test = time_features(test)
    print('time_features', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()

    # веду колонки которые просуммирую при группировке кредитов
    sum_cols = []
    train, sum_cols = features_1(train)
    test, _ = features_1(test)
    sum_cols2 = ['CREDIT_COLLATERAL', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT',
                 'AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_SUM_LIMIT', 'CREDIT_DELAY30', 'CREDIT_DELAY5',
                 'CREDIT_DELAY60', 'CREDIT_DELAY90', 'CREDIT_DELAY_MORE', 'AMT_REQ_SOURCE_YEAR', 'AMT_ANNUITY']
    max_cols = ['CREDIT_DAY_OVERDUE', 'AMT_CREDIT_MAX_OVERDUE', 'AMT_REQ_SOURCE_HOUR', 'AMT_REQ_SOURCE_DAY',
                'AMT_REQ_SOURCE_WEEK', 'AMT_REQ_SOURCE_MON', 'AMT_REQ_SOURCE_QRT']
    max_cols.extend(['worst_status_3m', 'worst_status_6m', 'worst_status_12m', 'worst_status_24m', 'worst_status_36m',
                     'worst_status_48m', 'worst_status_60m', 'lastcredit_time'])
    mean_cols = ['preschedule_time', 'plan_lifetime', 'fact_lifetime']
    min_cols = ['plan_lifetime', 'fact_lifetime', 'lastcredit_time', 'lastclosed_time']
    agg_col = []
    print('features_1', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()

    # аггрегаты
    df_train = agg_simple(train, [sum_cols, sum_cols2, max_cols, mean_cols, min_cols])
    df_test = agg_simple(test, [sum_cols, sum_cols2, max_cols, mean_cols, min_cols])
    print('agg_simple', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()

    # кросс
    df_train = cross_features(df_train, train)
    df_test = cross_features(df_test, test)
    print('cross', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()

    # двойной кросс
    df_train = double_cross_features(df_train, train)
    df_test = double_cross_features(df_test, test)
    print('double cross', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()

    ids_train = pd.DataFrame(train.groupby(['ID'])['SK_DATE_DECISION'].max().index)
    ids_test = pd.DataFrame(test.groupby(['ID'])['SK_DATE_DECISION'].max().index)
    target = train.groupby(['ID'])['DEF'].max().reset_index(drop=True)

    df_train.to_pickle('preprocess_features/train.pkl', compression='gzip')
    df_test.to_pickle('preprocess_features/test.pkl', compression='gzip')
    ids_train.to_pickle('preprocess_features/ids_train.pkl', compression='gzip')
    ids_test.to_pickle('preprocess_features/ids_test.pkl', compression='gzip')
    target.to_pickle('preprocess_features/target.pkl', compression='gzip')

    print('pickle', timeit.default_timer() - start_time)

    print('total', timeit.default_timer() - start_time1)
