# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from params import col_nuls,col_less_10, RANDOM_STATE, params_xgb,params_lgb

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
	train = train[list(set(train.columns.values) - set(col_nuls) - set(col_less_10))]
	test = test[list(set(test.columns.values) - set(col_nuls) - set(col_less_10))]

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

		xgb_test_global = lgb.Dataset(test.reindex(columns=X_train.columns), feature_name='auto')
		xgb_train = lgb.Dataset(X_train, label=y_train['DEF'], feature_name='auto')
		xgb_test = lgb.Dataset(X_test, label=y_test['DEF'], feature_name='auto')

		eval_result = {}

		model_fit = lgb.train(params_lgb, xgb_train, num_boost_round=1600,
				  valid_sets=[xgb_train, xgb_test],
				  valid_names=['train', 'eval'],
				  evals_result=eval_result,
				  verbose_eval=False)

		scores_train.loc[X_test.index, 'lgb_predict'] = model_fit.predict(X_test)
		scores_test['lgb_predict_'+str(fold)] = model_fit.predict(test.reindex(columns=X_train.columns))

		scores[fold] = eval_result['eval']['auc']

	scores = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in scores.items()]))

	std = scores.std(axis=1)
	sco = scores.mean(axis=1)
	print(np.argmax(sco), sco.max(), std[np.argmax(sco)])
	scores.to_pickle('data/folds/scores.pkl', compression='gzip')
	scores_train.to_pickle('data/stacking/scores_train_lgb.pkl', compression='gzip')
	# pd.DataFrame(scores_test.mean(axis=1), columns=['xgb_predict']).to_pickle('data/stacking/scores_test_lgb.pkl', compression='gzip')
	scores_test.to_pickle('data/stacking/scores_test_lgb.pkl', compression='gzip')


#923 0.68957220103 0.00954397423582
#public 0.696414

#1130 0.689657610337 0.0101409758008