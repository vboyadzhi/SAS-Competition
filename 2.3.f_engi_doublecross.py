# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import timeit
from itertools import product



def double_cross_features(df_source):
	def groupby_col(df_func, col_groupby, col_value, agg):
		func = {'sum': sum, 'count': np.count_nonzero, 'min': min, 'max': max, 'mean': np.mean}[agg]
		col_groupby1, col_groupby2 = col_groupby
		temp = df_func[['ID', col_value, col_groupby1, col_groupby2]].groupby(
			['ID', col_groupby1, col_groupby2]).agg(func).reset_index()
		temp['cols'] = temp[col_groupby1].astype(str) + '_' + temp[col_groupby2].astype(str)
		temp = temp.pivot(index='ID', columns='cols', values=col_value)
		temp.columns = temp.columns.map(
			lambda x: agg + '_' + col_groupby1 + '_' + col_groupby2 + '_' + col_value + '_' + str(x))
		return temp

	return pd.concat([
					groupby_col(df_source, ['CREDIT_ACTIVE', 'CREDIT_TYPE'], 'AMT_CREDIT_SUM', 'min'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_TYPE'], 'AMT_CREDIT_SUM', 'min'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_ACTIVE'], 'AMT_CREDIT_SUM', 'min'),
					groupby_col(df_source, ['CREDIT_ACTIVE', 'CREDIT_TYPE'], 'AMT_CREDIT_SUM', 'mean'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_TYPE'], 'AMT_CREDIT_SUM', 'mean'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_ACTIVE'], 'AMT_CREDIT_SUM', 'mean'),
					groupby_col(df_source, ['CREDIT_ACTIVE', 'CREDIT_TYPE'], 'AMT_CREDIT_SUM_DEBT', 'min'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_TYPE'], 'AMT_CREDIT_SUM_DEBT', 'min'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_ACTIVE'], 'AMT_CREDIT_SUM_DEBT', 'min'),
					groupby_col(df_source, ['CREDIT_ACTIVE', 'CREDIT_TYPE'], 'AMT_CREDIT_SUM_DEBT', 'mean'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_TYPE'], 'AMT_CREDIT_SUM_DEBT', 'mean'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_ACTIVE'], 'AMT_CREDIT_SUM_DEBT', 'mean'),
					groupby_col(df_source, ['CREDIT_ACTIVE', 'CREDIT_TYPE'], 'lastcredit_time', 'min'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_TYPE'], 'lastcredit_time', 'min'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_ACTIVE'], 'lastcredit_time', 'min'),
					groupby_col(df_source, ['CREDIT_ACTIVE', 'CREDIT_TYPE'], 'lastcredit_time', 'mean'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_TYPE'], 'lastcredit_time', 'mean'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_ACTIVE'], 'lastcredit_time', 'mean'),
					groupby_col(df_source, ['CREDIT_ACTIVE', 'CREDIT_TYPE'], 'plan_lifetime', 'min'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_TYPE'], 'plan_lifetime', 'min'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_ACTIVE'], 'plan_lifetime', 'min'),
					groupby_col(df_source, ['CREDIT_ACTIVE', 'CREDIT_TYPE'], 'plan_lifetime', 'mean'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_TYPE'], 'plan_lifetime', 'mean'),
					groupby_col(df_source, ['CREDIT_CURRENCY', 'CREDIT_ACTIVE'], 'plan_lifetime', 'mean'),
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

	# двойной кросс
	union = pd.concat([train, test], axis=0)
	union = double_cross_features(union)
	train = union.loc[ids_train]
	test = union.loc[ids_test]
	print('double cross', timeit.default_timer() - start_time)
	start_time = timeit.default_timer()

	train.to_pickle('data/double_cross/train.pkl', compression='gzip')
	test.to_pickle('data/double_cross/test.pkl', compression='gzip')

	print('pickle', timeit.default_timer() - start_time)

	print('total', timeit.default_timer() - start_time1)
