import logging
import pandas as pd
import xgboost as xgb
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from params import col_nuls,col_less_10, RANDOM_STATE, params_xgb


space = {
	'eta': 0.01,
	'max_depth': hp.quniform('max_depth', 3, 20, 1),
	'min_child_weight': hp.quniform('min_child_weight', 1, 7, 1),
	'subsample': hp.quniform('subsample', 0.6, 1, 0.1),
	'gamma': hp.quniform('gamma', 0, 0.1, 0.01),
	'alpha': hp.quniform('alpha', 0, 1, 0.1),
	'lambda': hp.quniform('lambda', 0, 1, 0.1),
	'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 1, 0.1),
	'objective': 'binary:logistic',
	'eval_metric': 'auc',
	'booster': 'gbtree',
	'seed': RANDOM_STATE
}


def score(params):
	global xgb_train
	# seed = int(np.random.rand()*100000)
	params['max_depth'] = int(params['max_depth'])
	params['min_child_weight'] = int(params['min_child_weight'])
	logger.info('Training with params:')
	logger.info(params)
	cv_res = xgb.cv(params, xgb_train, num_boost_round=2000, nfold=5, stratified=True,early_stopping_rounds=30)
	score = np.max(cv_res['test-auc-mean'])
	logger.info('score = %f' % score)
	logger.info('best_rounds = %f' % np.argmax(cv_res['test-auc-mean']))

	return {'loss': -score, 'status': STATUS_OK}

if __name__ == "__main__":
	logger = logging.getLogger('server_logger')
	logger.setLevel(logging.INFO)
	# create file handler which logs even debug messages
	fh = logging.FileHandler('server_xgb.log')
	fh.setLevel(logging.INFO)
	# create console handler with a higher log level
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	# create formatter and add it to the handlers
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
	ch.setFormatter(formatter)
	fh.setFormatter(formatter)
	# add the handlers to logger
	logger.addHandler(ch)
	logger.addHandler(fh)
	# logger.info('This should go to both console and file')


	# test = pd.concat([pd.read_pickle('data/aggregation/test.pkl', compression='gzip'),
	# 				pd.read_pickle('data/cross/test.pkl', compression='gzip'),
	# 				pd.read_pickle('data/double_cross/test.pkl', compression='gzip')],axis=1)
	# ids_test = pd.read_pickle('data/drop_duplicates/ids_test.pkl', compression='gzip')

	train = pd.concat([pd.read_pickle('data/aggregation/train.pkl', compression='gzip'),
					   pd.read_pickle('data/cross/train.pkl', compression='gzip'),
					   pd.read_pickle('data/double_cross/train.pkl', compression='gzip')], axis=1)

	ids_train = pd.read_pickle('data/drop_duplicates/ids_train.pkl', compression='gzip')
	target = pd.read_pickle('data/drop_duplicates/target.pkl', compression='gzip')

	train = train[list(set(train.columns.values) - set(col_nuls) - set(col_less_10))]

	xgb_train = xgb.DMatrix(train, label=target)

	trials = Trials()
	best = fmin(fn=score,
				space=space,
				algo=tpe.suggest,
				trials=trials,
				max_evals=150
				)



'''
2017-08-31 15:22:56 - INFO - Training with params:
2017-08-31 15:22:56 - INFO - {'colsample_bytree': 1.0, 'eval_metric': 'auc', 'min_child_weight': 6, 'subsample': 0.6000000000000001, 'eta': 0.01, 'objective': 'binary:logistic', 'alpha': 0.9, 'booster': 'gbtree', 'seed': 2017, 'max_depth': 4, 'gamma': 0.05, 'lambda': 0.6000000000000001}
2017-08-31 15:42:43 - INFO - score = 0.701148
2017-08-31 15:42:43 - INFO - best_rounds = 1296.000000        
2017-08-31 16:06:34 - INFO - Training with params:
2017-08-31 16:06:34 - INFO - {'colsample_bytree': 1.0, 'eval_metric': 'auc', 'min_child_weight': 6, 'subsample': 0.6000000000000001, 'eta': 0.01, 'objective': 'binary:logistic', 'alpha': 0.5, 'booster': 'gbtree', 'seed': 2017, 'max_depth': 7, 'gamma': 0.01, 'lambda': 0.6000000000000001}
2017-08-31 16:32:25 - INFO - score = 0.700087
2017-08-31 16:32:25 - INFO - best_rounds = 970.000000
2017-08-31 16:32:25 - INFO - Training with params:
2017-08-31 16:32:25 - INFO - {'colsample_bytree': 0.9, 'eval_metric': 'auc', 'min_child_weight': 6, 'subsample': 0.6000000000000001, 'eta': 0.01, 'objective': 'binary:logistic', 'alpha': 0.4, 'booster': 'gbtree', 'seed': 2017, 'max_depth': 4, 'gamma': 0.03, 'lambda': 0.7000000000000001}
2017-08-31 16:50:31 - INFO - score = 0.700914
2017-08-31 16:50:31 - INFO - best_rounds = 1294.000000
2017-08-31 17:44:39 - INFO - Training with params:
2017-08-31 17:44:39 - INFO - {'colsample_bytree': 1.0, 'eval_metric': 'auc', 'min_child_weight': 7, 'subsample': 0.7000000000000001, 'eta': 0.01, 'objective': 'binary:logistic', 'alpha': 1.0, 'booster': 'gbtree', 'seed': 2017, 'max_depth': 5, 'gamma': 0.01, 'lambda': 0.4}
2017-08-31 18:06:31 - INFO - score = 0.700996
2017-08-31 18:06:31 - INFO - best_rounds = 1160.000000
'''