# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from params import col_nuls,col_less_10, RANDOM_STATE, params_xgb



if __name__ == "__main__":
	# test = pd.concat([pd.read_pickle('data/aggregation/test.pkl', compression='gzip'),
	# 				pd.read_pickle('data/cross/test.pkl', compression='gzip'),
	# 				pd.read_pickle('data/double_cross/test.pkl', compression='gzip')],axis=1)
	# ids_test = pd.read_pickle('data/drop_duplicates/ids_test.pkl', compression='gzip')

	train = pd.concat([pd.read_pickle('data/aggregation/train.pkl', compression='gzip'),
					pd.read_pickle('data/cross/train.pkl', compression='gzip'),
					pd.read_pickle('data/double_cross/train.pkl', compression='gzip')],axis=1)

	#ids_train = pd.read_pickle('data/drop_duplicates/ids_train.pkl', compression='gzip')
	target = pd.read_pickle('data/drop_duplicates/target.pkl', compression='gzip')

	# train = train[list(set(train.columns.values) - set(col_nuls) - set(col_less_10))]
	train = train[list(set(train.columns.values))]

	xgb_train = xgb.DMatrix(train, label=target)
	cv_res = xgb.cv(params_xgb, xgb_train, num_boost_round=1500, nfold=5, stratified=True, )
	num_boost_best = np.argmax(cv_res['test-auc-mean'])
	print (np.argmax(cv_res['test-auc-mean']), np.max(cv_res['test-auc-mean']))
#988 0.7000424
#1052 0.7017934 + id