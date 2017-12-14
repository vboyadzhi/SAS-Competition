# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import timeit
from itertools import product




def cross_features(df_source):
    def groupby_col(df_func, col_groupby, col_value, agg):
        func = {'sum': sum, 'count': np.count_nonzero, 'min': min, 'max': max, 'mean': np.mean}[agg]
        temp = df_func[['ID', col_groupby, col_value]].groupby(['ID', col_groupby]).agg(func) \
            .reset_index().pivot(index='ID', columns=col_groupby, values=col_value)
        temp.columns = temp.columns.map(lambda x: agg + '_' + col_groupby + '_' + col_value + '_' + str(x))
        return temp

    # кросс min
    return pd.concat([

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'plan_lifetime', 'min'),
                    groupby_col(df_source, 'CREDIT_CURRENCY', 'plan_lifetime', 'min'),
                    groupby_col(df_source, 'CREDIT_TYPE', 'plan_lifetime', 'min'),
                    groupby_col(df_source, 'CREDIT_SUM_TYPE', 'plan_lifetime', 'min'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'plan_lifetime', 'max'),
                    groupby_col(df_source, 'CREDIT_CURRENCY', 'plan_lifetime', 'max'),
                    groupby_col(df_source, 'CREDIT_TYPE', 'plan_lifetime', 'max'),
                    groupby_col(df_source, 'CREDIT_SUM_TYPE', 'plan_lifetime', 'max'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'plan_lifetime', 'mean'),
                    groupby_col(df_source, 'CREDIT_CURRENCY', 'plan_lifetime', 'mean'),
                    groupby_col(df_source, 'CREDIT_TYPE', 'plan_lifetime', 'mean'),
                    groupby_col(df_source, 'CREDIT_SUM_TYPE', 'plan_lifetime', 'mean'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'lastupdate_time', 'min'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'lastupdate_time', 'min'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'lastupdate_time', 'min'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'lastupdate_time', 'max'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'lastupdate_time', 'max'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'lastupdate_time', 'max'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'lastupdate_time', 'mean'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'lastupdate_time', 'mean'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'lastupdate_time', 'mean'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'lastcredit_time', 'min'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'lastcredit_time', 'min'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'lastcredit_time', 'min'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'lastcredit_time', 'max'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'lastcredit_time', 'max'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'lastcredit_time', 'max'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'lastcredit_time', 'mean'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'lastcredit_time', 'mean'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'lastcredit_time', 'mean'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'preschedule_time', 'min'),
                    groupby_col(df_source, 'CREDIT_CURRENCY', 'preschedule_time', 'min'),
                    groupby_col(df_source, 'CREDIT_TYPE', 'preschedule_time', 'min'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'preschedule_time', 'max'),
                    groupby_col(df_source, 'CREDIT_CURRENCY', 'preschedule_time', 'max'),
                    groupby_col(df_source, 'CREDIT_TYPE', 'preschedule_time', 'max'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'preschedule_time', 'mean'),
                    groupby_col(df_source, 'CREDIT_CURRENCY', 'preschedule_time', 'mean'),
                    groupby_col(df_source, 'CREDIT_TYPE', 'preschedule_time', 'mean'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'fact_lifetime', 'min'),
                    groupby_col(df_source, 'CREDIT_CURRENCY', 'fact_lifetime', 'min'),
                    groupby_col(df_source, 'CREDIT_TYPE', 'fact_lifetime', 'min'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'fact_lifetime', 'max'),
                    groupby_col(df_source, 'CREDIT_CURRENCY', 'fact_lifetime', 'max'),
                    groupby_col(df_source, 'CREDIT_TYPE', 'fact_lifetime', 'max'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'fact_lifetime', 'mean'),
                    groupby_col(df_source, 'CREDIT_CURRENCY', 'fact_lifetime', 'mean'),
                    groupby_col(df_source, 'CREDIT_TYPE', 'fact_lifetime', 'mean'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_MAX_OVERDUE', 'min'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_MAX_OVERDUE', 'min'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_MAX_OVERDUE', 'min'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_MAX_OVERDUE', 'max'),
                    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_MAX_OVERDUE', 'max'),
                    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_MAX_OVERDUE', 'max'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_MAX_OVERDUE', 'mean'),
                    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_MAX_OVERDUE', 'mean'),
                    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_MAX_OVERDUE', 'mean'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_MAX_OVERDUE', 'sum'),
                    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_MAX_OVERDUE', 'sum'),
                    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_MAX_OVERDUE', 'sum'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM', 'min'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_SUM', 'min'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_SUM', 'min'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM', 'max'),
                    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_SUM', 'max'),
                    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_SUM', 'max'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM', 'mean'),
                    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_SUM', 'mean'),
                    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_SUM', 'mean'),

                    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM', 'sum'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_SUM', 'sum'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_SUM', 'sum'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_ANNUITY', 'min'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_ANNUITY', 'min'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_ANNUITY', 'min'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_ANNUITY', 'max'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_ANNUITY', 'max'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_ANNUITY', 'max'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_ANNUITY', 'mean'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_ANNUITY', 'mean'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_ANNUITY', 'mean'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_ANNUITY', 'sum'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_ANNUITY', 'sum'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_ANNUITY', 'sum'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM_DEBT', 'min'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_SUM_DEBT', 'min'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_SUM_DEBT', 'min'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM_DEBT', 'max'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_SUM_DEBT', 'max'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_SUM_DEBT', 'max'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM_DEBT', 'mean'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_SUM_DEBT', 'mean'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_SUM_DEBT', 'mean'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM_DEBT', 'sum'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_SUM_DEBT', 'sum'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_SUM_DEBT', 'sum'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM_LIMIT', 'min'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_SUM_LIMIT', 'min'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_SUM_LIMIT', 'min'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM_LIMIT', 'max'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_SUM_LIMIT', 'max'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_SUM_LIMIT', 'max'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM_LIMIT', 'mean'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_SUM_LIMIT', 'mean'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_SUM_LIMIT', 'mean'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM_LIMIT', 'sum'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'AMT_CREDIT_SUM_LIMIT', 'sum'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'AMT_CREDIT_SUM_LIMIT', 'sum'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'ones_0:3', 'min'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'ones_0:3', 'min'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'ones_0:3', 'min'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'ones_0:3', 'max'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'ones_0:3', 'max'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'ones_0:3', 'max'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'ones_0:3', 'mean'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'ones_0:3', 'mean'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'ones_0:3', 'mean'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'ones_0:3', 'sum'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'ones_0:3', 'sum'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'ones_0:3', 'sum'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'ones_0:6', 'min'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'ones_0:6', 'min'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'ones_0:6', 'min'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'ones_0:6', 'max'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'ones_0:6', 'max'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'ones_0:6', 'max'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'ones_0:6', 'mean'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'ones_0:6', 'mean'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'ones_0:6', 'mean'),

				    groupby_col(df_source, 'CREDIT_ACTIVE', 'ones_0:6', 'sum'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'ones_0:6', 'sum'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'ones_0:6', 'sum'),

				    # Количество
				    groupby_col(df_source, 'CREDIT_ACTIVE', 'SK_DATE_DECISION', 'count'),
				    groupby_col(df_source, 'CREDIT_CURRENCY', 'SK_DATE_DECISION', 'count'),
				    groupby_col(df_source, 'CREDIT_TYPE', 'SK_DATE_DECISION', 'count'),
    ], axis=1)


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

    # кросс
    union = pd.concat([train, test], axis=0)
    union = cross_features(union)
    train = union.loc[ids_train]
    test = union.loc[ids_test]
    print('cross', timeit.default_timer() - start_time)
    start_time = timeit.default_timer()

    train.to_pickle('data/cross/train.pkl', compression='gzip')
    test.to_pickle('data/cross/test.pkl', compression='gzip')

    print('pickle', timeit.default_timer() - start_time)

    print('total', timeit.default_timer() - start_time1)
