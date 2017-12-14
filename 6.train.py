# -*- coding: utf-8 -*-
import pandas as pd
import xgboost as xgb
import bz2
import _pickle as pickle
from params import col_nuls,col_less_10, RANDOM_STATE, params_xgb

if __name__ == "__main__":
	# test = pd.concat([pd.read_pickle('data/aggregation/test.pkl', compression='gzip'),
	# 				pd.read_pickle('data/cross/test.pkl', compression='gzip'),
	# 				pd.read_pickle('data/double_cross/test.pkl', compression='gzip')],axis=1)
	# ids_test = pd.read_pickle('data/drop_duplicates/ids_test.pkl', compression='gzip')

	train = pd.concat([pd.read_pickle('data/aggregation/train.pkl', compression='gzip'),
					pd.read_pickle('data/cross/train.pkl', compression='gzip'),
					pd.read_pickle('data/double_cross/train.pkl', compression='gzip'),
					pd.concat([pd.read_pickle('data/folds/X_test_fold_0.csv', compression='gzip'),
								   pd.read_pickle('data/folds/X_test_fold_1.csv', compression='gzip'),
								   pd.read_pickle('data/folds/X_test_fold_2.csv', compression='gzip'),
								   pd.read_pickle('data/folds/X_test_fold_3.csv', compression='gzip'),
								   pd.read_pickle('data/folds/X_test_fold_4.csv', compression='gzip')],
								axis=0)],
					axis = 1)

	ids_train = pd.read_pickle('data/drop_duplicates/ids_train.pkl', compression='gzip')

	target = pd.read_pickle('data/drop_duplicates/target.pkl', compression='gzip')

	#train = train[list(set(train.columns.values) - set(col_nuls) - set(col_less_10))]

	xgb_train = xgb.DMatrix(train, label=target)

	num_boost_best = 1855
	model_fit = xgb.train(params_xgb, xgb_train,
						  num_boost_round=num_boost_best,
						  # evals = watchlist,
						  maximize=True,
						  verbose_eval=False
						  )
	with bz2.BZ2File('models/model_xgb.pbz2', 'w') as f:
		pickle.dump(model_fit, f)



