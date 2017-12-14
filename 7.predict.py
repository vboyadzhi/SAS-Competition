# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
import bz2
import _pickle as pickle

RANDOM_STATE = 0

if __name__ == "__main__":
	test = pd.concat([pd.read_pickle('data/aggregation/test.pkl', compression='gzip'),
					pd.read_pickle('data/cross/test.pkl', compression='gzip'),
					pd.read_pickle('data/double_cross/test.pkl', compression='gzip'),
					pd.read_pickle('data/folds/test_mean_target.csv', compression='gzip')],
	axis=1)

	# ids_test = pd.read_pickle('data/drop_duplicates/ids_test.pkl', compression='gzip')
	# target = pd.read_pickle('data/drop_duplicates/target.pkl', compression='gzip')


	# train = pd.concat([pd.read_pickle('data/aggregation/train.pkl', compression='gzip'),
	# 				pd.read_pickle('data/cross/train.pkl', compression='gzip'),
	# 				pd.read_pickle('data/double_cross/train.pkl', compression='gzip')],axis=1)
	# ids_train = pd.read_pickle('data/drop_duplicates/ids_train.pkl', compression='gzip')
	# train = train[list(set(train.columns.values) - set(col_nuls) - set(col_less_10))]

	with bz2.BZ2File('models/model_xgb.pbz2', 'r') as f:
		model_fit = pickle.load(f)

	xgb_test = xgb.DMatrix(test.reindex(columns=model_fit.feature_names))

	test['Score'] = model_fit.predict(xgb_test)
	test.reset_index()[['ID', 'Score']].to_csv('submission/sub_xgb.csv', index=False)
