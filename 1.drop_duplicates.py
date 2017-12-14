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
    mask = df[['ID', 'NUM_SOURCE', 'CREDIT_CURRENCY', 'DTIME_CREDIT', 'AMT_CREDIT_SUM', 'CREDIT_TYPE']] \
        .sort_values(['ID', 'DTIME_CREDIT', 'NUM_SOURCE'], ascending=[True, False, False]) \
        .drop('NUM_SOURCE', axis=1) \
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
    df['zeros'] = df['TEXT_PAYMENT_DISCIPLINE'].str.replace('C', 'q').str.replace('X', 'q').str.replace('1', 'q') \
        .str.replace('2', 'q').str.replace('3', 'q').str.replace('4', 'q').str.replace('5', 'q')
    df['ones'] = df['TEXT_PAYMENT_DISCIPLINE'].str.replace('C', 'q').str.replace('X', 'q').str.replace('0', 'q') \
        .str.replace('2', 'q').str.replace('3', 'q').str.replace('4', 'q').str.replace('5', 'q')
    df['twos'] = df['TEXT_PAYMENT_DISCIPLINE'].str.replace('C', 'q').str.replace('X', 'q').str.replace('1', 'q') \
        .str.replace('0', 'q').str.replace('3', 'q').str.replace('4', 'q').str.replace('5', 'q')
    df['threes'] = df['TEXT_PAYMENT_DISCIPLINE'].str.replace('C', 'q').str.replace('X', 'q').str.replace('1', 'q') \
        .str.replace('2', 'q').str.replace('0', 'q').str.replace('4', 'q').str.replace('5', 'q')
    df['fours'] = df['TEXT_PAYMENT_DISCIPLINE'].str.replace('C', 'q').str.replace('X', 'q').str.replace('1', 'q') \
        .str.replace('2', 'q').str.replace('3', 'q').str.replace('0', 'q').str.replace('5', 'q')
    df['fives'] = df['TEXT_PAYMENT_DISCIPLINE'].str.replace('C', 'q').str.replace('X', 'q').str.replace('1', 'q') \
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
        cols.append(col)
    return df, cols


def time_features(df):
    df['plan_lifetime'] = (df['DTIME_CREDIT_ENDDATE'] - df['DTIME_CREDIT']).map(
        lambda x: x.days if x == x else np.nan) / 365
    df['lastclosedplan_time'] = (df['SK_DATE_DECISION'] - df['DTIME_CREDIT_ENDDATE']).map(
        lambda x: x.days if x == x else np.nan) / 365
    df['preschedule_time'] = (df['DTIME_CREDIT_ENDDATE'] - df['DTIME_CREDIT_ENDDATE_FACT']).map(
        lambda x: x.days if x == x else np.nan) / 365
    df['end_update_time'] = (df['DTIME_CREDIT_ENDDATE'] - df['DTIME_CREDIT_UPDATE']).map(
        lambda x: x.days if x == x else np.nan) / 365
    df['fact_lifetime'] = (df['DTIME_CREDIT_ENDDATE_FACT'] - df['DTIME_CREDIT']).map(
        lambda x: x.days if x == x else np.nan) / 365
    df['lastclosed_time'] = (df['SK_DATE_DECISION'] - df['DTIME_CREDIT_ENDDATE_FACT']).map(
        lambda x: x.days if x == x else np.nan) / 365
    df['endfact_update_time'] = (df['DTIME_CREDIT_ENDDATE_FACT'] - df['DTIME_CREDIT_UPDATE']).map(
        lambda x: x.days if x == x else np.nan) / 365
    df['lastupdate_time'] = (df['SK_DATE_DECISION'] - df['DTIME_CREDIT_UPDATE']).map(
        lambda x: x.days if x == x else np.nan) / 365
    df['lastcredit_time'] = (df['SK_DATE_DECISION'] - df['DTIME_CREDIT']).map(
        lambda x: x.days if x == x else np.nan) / 365
    df['credit_update_time'] = (df['DTIME_CREDIT'] - df['DTIME_CREDIT_UPDATE']).map(
        lambda x: x.days if x == x else np.nan) / 365
    return df

def currency(df):
    mask = ~(df.CREDIT_CURRENCY == 'rur')
    for col in ['AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_MAX_OVERDUE', 'AMT_ANNUITY']:
        df.loc[mask, col] = df.loc[mask, col]*60
    return df

if __name__ == "__main__":
    RANDOM_STATE = 0
    print('Start')

    start_time1 = timeit.default_timer()
    start_time = timeit.default_timer()
    test = pd.read_csv('data/raw/test.csv', sep=',')
    train = pd.read_csv('data/raw/train.csv', sep=',')
    print('read data', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()

    train = currency(train)
    test = currency(test)
    print('currency', timeit.default_timer() - start_time)

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
    print('features_1', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()

    train['CREDIT_FACILITY'] = train['CREDIT_FACILITY'] != 9
    test['CREDIT_FACILITY'] = test['CREDIT_FACILITY'] != 9

    ids_train = pd.Series(train.groupby(['ID'])['SK_DATE_DECISION'].max().index)
    ids_test = pd.Series(test.groupby(['ID'])['SK_DATE_DECISION'].max().index)
    target = train.groupby(['ID'])['DEF'].max()

    ids_train.to_pickle('data/drop_duplicates/ids_train.pkl', compression='gzip')
    ids_test.to_pickle('data/drop_duplicates/ids_test.pkl', compression='gzip')
    target.to_pickle('data/drop_duplicates/target.pkl', compression='gzip')
    train.to_pickle('data/drop_duplicates/train.pkl', compression='gzip')
    test.to_pickle('data/drop_duplicates/test.pkl', compression='gzip')
    pd.Series(sum_cols, name='sum_cols').to_pickle('data/drop_duplicates/sum_cols.pkl', compression='gzip')
    print('pickle', timeit.default_timer() - start_time)

    print('total', timeit.default_timer() - start_time1)
