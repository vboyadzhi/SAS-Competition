# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
import bz2
import _pickle as pickle
from params import col_nuls,col_less_10, RANDOM_STATE, params_xgb

if __name__ == "__main__":
	test = pd.concat([pd.read_pickle('data/aggregation/test.pkl', compression='gzip'),
					pd.read_pickle('data/cross/test.pkl', compression='gzip'),
					pd.read_pickle('data/double_cross/test.pkl', compression='gzip'),
					pd.read_pickle('data/folds/test_mean_target.csv', compression='gzip')],axis=1)
	# ids_test = pd.read_pickle('data/drop_duplicates/ids_test.pkl', compression='gzip')

	train = pd.concat([pd.read_pickle('data/aggregation/train.pkl', compression='gzip'),
					pd.read_pickle('data/cross/train.pkl', compression='gzip'),
					pd.read_pickle('data/double_cross/train.pkl', compression='gzip')],axis=1)

	# ids_train = pd.read_pickle('data/drop_duplicates/ids_train.pkl', compression='gzip')
	# target = pd.read_pickle('data/drop_duplicates/target.pkl', compression='gzip')
	#
	# train = train[list(set(train.columns.values) - set(col_nuls) - set(col_less_10))]
	# test = test[list(set(test.columns.values) - set(col_nuls) - set(col_less_10))]

	scores_train = pd.DataFrame(index = train.index)
	scores_test = pd.DataFrame(index = test.index)

	print('cross validation')

	SPLITS = 5
	scores = {}
	for fold in np.arange(SPLITS):

		X_train = pd.read_pickle('data/folds/X_train_fold_' + str(fold) + '.csv', compression='gzip')
		X_test = pd.read_pickle('data/folds/X_test_fold_' + str(fold) + '.csv', compression='gzip')
		y_train = pd.read_pickle('data/folds/y_train_fold_' + str(fold) + '.csv', compression='gzip')
		y_test = pd.read_pickle('data/folds/y_test_fold_' + str(fold) + '.csv', compression='gzip')

		X_train = pd.merge(train, X_train, how='inner', left_index=True, right_index=True)
		X_test = pd.merge(train, X_test, how='inner', left_index=True, right_index=True)

		xgb_test_global = xgb.DMatrix(test.reindex(columns=X_train.columns))
		xgb_train = xgb.DMatrix(X_train, label=y_train)
		xgb_test = xgb.DMatrix(X_test, label=y_test)

		watchlist = [(xgb_train, 'train'), (xgb_test, 'eval')]

		eval_result = {}

		model_fit = xgb.train(params_xgb, xgb_train, num_boost_round=2000,
							  evals=watchlist,
							  maximize=False,
							  verbose_eval=False,
							  #early_stopping_rounds = 200,
							  callbacks=[xgb.callback.record_evaluation(eval_result)]
							  )

		with bz2.BZ2File('models/model_xgb_fold'+str(fold)+'.pbz2', 'w') as f:
			pickle.dump(model_fit, f)

		scores_train.loc[X_test.index, 'xgb_predict'] = model_fit.predict(xgb_test)
		scores_test['xgb_predict_'+str(fold)] = model_fit.predict(xgb_test_global)

		scores[fold] = eval_result['eval']['auc']

	scores = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in scores.items()]))

	std = scores.std(axis=1)
	sco = scores.mean(axis=1)
	print(np.argmax(sco), sco.max(), std[np.argmax(sco)])
	scores.to_pickle('data/folds/scores.pkl', compression='gzip')
	scores_train.to_pickle('data/stacking/scores_train.pkl', compression='gzip')
	#pd.DataFrame(scores_test.mean(axis=1), columns=['xgb_predict']).to_pickle('data/stacking/scores_test.pkl', compression='gzip')
	scores_test.to_pickle('data/stacking/scores_test.pkl', compression='gzip')

#1526 0.7001176 0.00734083839217
#public 0.709877
#1855 0.7012344 0.00869099777356
#public 0.711399