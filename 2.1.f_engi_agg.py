# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import timeit
from itertools import product


def agg_simple(df, cols):
    sum_cols, sum_cols2, max_cols, mean_cols, min_cols = cols
    x = df.groupby(['ID'])[sum_cols].sum()
    x.columns = x.columns.map(lambda y: 'sum' + '_' + str(y))
    x2 = df.groupby(['ID'])[sum_cols2].sum()
    x2.columns = x2.columns.map(lambda y: 'sum' + '_' + str(y))
    x3 = df.groupby(['ID'])[max_cols].max()
    x3.columns = x3.columns.map(lambda y: 'max' + '_' + str(y))
    x4 = df.groupby(['ID'])[mean_cols].mean()
    x4.columns = x4.columns.map(lambda y: 'mean' + '_' + str(y))
    x5 = df.groupby(['ID'])[min_cols].mean()
    x5.columns = x5.columns.map(lambda y: 'min' + '_' + str(y))
    x6 = df.groupby(['ID'])['SK_DATE_DECISION'].count()
    x6.name = 'count'
    return pd.concat([x, x2, x3, x4, x5, x6], axis=1)


if __name__ == "__main__":
    RANDOM_STATE = 2017
    print('Start')

    start_time1 = timeit.default_timer()
    start_time = timeit.default_timer()
    test = pd.read_pickle('data/drop_duplicates/test.pkl', compression='gzip')
    train = pd.read_pickle('data/drop_duplicates/train.pkl', compression='gzip')
    ids_test = pd.read_pickle('data/drop_duplicates/ids_test.pkl', compression='gzip')
    ids_train = pd.read_pickle('data/drop_duplicates/ids_train.pkl', compression='gzip')
    print('read data', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()
    sum_cols = ['zeros_0:3',
                 'zeros_0:6',
                 'zeros_0:12',
                 'zeros_0:24',
                 'zeros_0:36',
                 'zeros_0:48',
                 'zeros_0:60',
                 'ones_0:3',
                 'ones_0:6',
                 'ones_0:12',
                 'ones_0:24',
                 'ones_0:36',
                 'ones_0:48',
                 'ones_0:60',
                 'twos_0:3',
                 'twos_0:6',
                 'twos_0:12',
                 'twos_0:24',
                 'twos_0:36',
                 'twos_0:48',
                 'twos_0:60',
                 'threes_0:3',
                 'threes_0:6',
                 'threes_0:12',
                 'threes_0:24',
                 'threes_0:36',
                 'threes_0:48',
                 'threes_0:60',
                 'fours_0:3',
                 'fours_0:6',
                 'fours_0:12',
                 'fours_0:24',
                 'fours_0:36',
                 'fours_0:48',
                 'fours_0:60',
                 'fives_0:3',
                 'fives_0:6',
                 'fives_0:12',
                 'fives_0:24',
                 'fives_0:36',
                 'fives_0:48',
                 'fives_0:60',
                 'bads_0:3',
                 'bads_0:6',
                 'bads_0:12',
                 'bads_0:24',
                 'bads_0:36',
                 'bads_0:48',
                 'bads_0:60',
                 'base_0:3',
                 'base_0:6',
                 'base_0:12',
                 'base_0:24',
                 'base_0:36',
                 'base_0:48',
                 'base_0:60',
                 'zeros',
                 'ones',
                 'twos',
                 'threes',
                 'fours',
                 'fives',
                 'bads',
                 'base']
    sum_cols2 = ['CREDIT_COLLATERAL',
                 'CNT_CREDIT_PROLONG',
                 'AMT_CREDIT_SUM',
                 'AMT_CREDIT_SUM_DEBT',
                 'AMT_CREDIT_SUM_OVERDUE',
                 'AMT_CREDIT_SUM_LIMIT',
                 'CREDIT_DELAY30',
                 'CREDIT_DELAY5',
                 'CREDIT_DELAY60',
                 'CREDIT_DELAY90',
                 'CREDIT_DELAY_MORE',
                 'AMT_ANNUITY',
                 'CREDIT_FACILITY']
    max_cols = ['CREDIT_DAY_OVERDUE',
                'AMT_CREDIT_MAX_OVERDUE',
                'AMT_REQ_SOURCE_HOUR',
                'AMT_REQ_SOURCE_DAY',
                'AMT_REQ_SOURCE_WEEK',
                'AMT_REQ_SOURCE_MON',
                'AMT_REQ_SOURCE_QRT',
                'AMT_REQ_SOURCE_YEAR']
    max_cols.extend(['worst_status_3m', 'worst_status_6m', 'worst_status_12m', 'worst_status_24m', 'worst_status_36m',
                     'worst_status_48m', 'worst_status_60m'])
    max_cols.extend(['plan_lifetime',
                     'lastclosedplan_time',
                     'preschedule_time',
                     'end_update_time',
                     'fact_lifetime',
                     'lastclosed_time',
                     'endfact_update_time',
                     'lastupdate_time',
                     'lastcredit_time',
                     'credit_update_time'])
    max_cols.extend(['CREDIT_COLLATERAL',
                     'CNT_CREDIT_PROLONG',
                     'AMT_CREDIT_SUM',
                     'AMT_CREDIT_SUM_DEBT',
                     'AMT_CREDIT_SUM_OVERDUE',
                     'AMT_CREDIT_SUM_LIMIT',
                     'CREDIT_DELAY30',
                     'CREDIT_DELAY5',
                     'CREDIT_DELAY60',
                     'CREDIT_DELAY90',
                     'CREDIT_DELAY_MORE',
                     'AMT_ANNUITY'])
    mean_cols = ['plan_lifetime',
                 'lastclosedplan_time',
                 'preschedule_time',
                 'end_update_time',
                 'fact_lifetime',
                 'lastclosed_time',
                 'endfact_update_time',
                 'lastupdate_time',
                 'lastcredit_time',
                 'credit_update_time']
    min_cols = ['plan_lifetime',
                'lastclosedplan_time',
                'preschedule_time',
                'end_update_time',
                'fact_lifetime',
                'lastclosed_time',
                'endfact_update_time',
                'lastupdate_time',
                'lastcredit_time',
                'credit_update_time']
    min_cols.extend(['CREDIT_DAY_OVERDUE',
                     'AMT_CREDIT_MAX_OVERDUE'])
    # sum_cols2 = ['CREDIT_COLLATERAL', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT',
    # 			 'AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_SUM_LIMIT', 'CREDIT_DELAY30', 'CREDIT_DELAY5',
    # 			 'CREDIT_DELAY60', 'CREDIT_DELAY90', 'CREDIT_DELAY_MORE', 'AMT_REQ_SOURCE_YEAR', 'AMT_ANNUITY']
    #
    # max_cols = ['CREDIT_DAY_OVERDUE', 'AMT_CREDIT_MAX_OVERDUE', 'AMT_REQ_SOURCE_HOUR', 'AMT_REQ_SOURCE_DAY',
    # 			'AMT_REQ_SOURCE_WEEK', 'AMT_REQ_SOURCE_MON', 'AMT_REQ_SOURCE_QRT']
    #
    # max_cols.extend(['worst_status_3m', 'worst_status_6m', 'worst_status_12m', 'worst_status_24m', 'worst_status_36m',
    # 				 'worst_status_48m', 'worst_status_60m', 'lastcredit_time'])
    #
    # mean_cols = ['preschedule_time', 'plan_lifetime', 'fact_lifetime']
    # min_cols = ['plan_lifetime', 'fact_lifetime', 'lastcredit_time', 'lastclosed_time']

    # аггрегаты
    union = pd.concat([train, test], axis=0)
    union = agg_simple(union, [sum_cols, sum_cols2, max_cols, mean_cols, min_cols])
    train = union.loc[ids_train]
    test = union.loc[ids_test]
    print('agg_simple', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()

    train.to_pickle('data/aggregation/train.pkl', compression='gzip')
    test.to_pickle('data/aggregation/test.pkl', compression='gzip')

    print('total', timeit.default_timer() - start_time1)
