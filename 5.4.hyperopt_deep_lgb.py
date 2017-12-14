import logging
import pandas as pd
import xgboost as xgb
import numpy as np
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from params import col_nuls,col_less_10, RANDOM_STATE, params_xgb


# space = {
#     'learning_rate' : 0.1,#hp.quniform('eta',0.001,0.1,0.001),
#     'num_leaves' : hp.quniform('num_leaves',300,1500,200),
#     'num_trees' : hp.quniform('num_trees',300,1500,200),
#     'min_data_in_leaf' : 0,#hp.quniform('subsample',0.5,1,0.1),
#     'min_sum_hessian_in_leaf' : 100,#hp.quniform('gamma',0,1.1,0.1),
#     'max_bin' : hp.quniform('max_bin',2,5,1),
#     'feature_fraction' : 0.9,
#     'boosting' : 'dart',
# 	'metric' : 'auc',
#     'num_threads' :4
# }
space ={
        #'boosting_type': hp.choice( 'boosting_type', ['gbdt', 'dart' ] ),
        #'max_depth': hp.quniform("max_depth", 4, 6, 1),
        'num_leaves': hp.quniform ('num_leaves', 20, 100, 1),
        'min_data_in_leaf':  hp.quniform ('min_data_in_leaf', 10, 100, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
        'learning_rate': 0.005,#hp.loguniform('learning_rate', -6.9, -2.3),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
        'lambda_l1': hp.uniform('lambda_l1', 1e-4, 1e-6 ),
        'lambda_l2': hp.uniform('lambda_l2', 1e-4, 1e-6 ),
        'seed': RANDOM_STATE
       }
def get_params(space):
    px = dict()

    px['boosting_type']='gbdt' # space['boosting_type'], # 'gbdt', # gbdt | dart | goss
    px['objective'] ='binary'
    px['metric'] = 'auc'
    px['learning_rate']=space['learning_rate']
    px['num_leaves'] = int(space['num_leaves'])
    px['min_data_in_leaf'] = int(space['min_data_in_leaf'])
    px['min_sum_hessian_in_leaf'] = space['min_sum_hessian_in_leaf']
    px['max_depth'] = int(space['max_depth']) if 'max_depth' in space else -1
    px['lambda_l1'] = space['lambda_l1'],
    px['lambda_l2'] = space['lambda_l2'],
    px['max_bin'] = 256
    px['feature_fraction'] = space['feature_fraction']
    px['bagging_fraction'] = space['bagging_fraction']
    px['bagging_freq'] = 5
    return px

def score(params):
	params['num_leaves'] = int(params['num_leaves'])
	# params['num_trees'] = int(params['num_trees'])
	# seed = int(np.random.rand()*100000)
	logger.info('Training with params:')
	logger.info(params)

	params = get_params(params)
	SPLITS = 5
	scores = {}
	for fold in np.arange(SPLITS):
		X_train = pd.read_pickle('data/folds/X_train_fold_' + str(fold) + '.csv', compression='gzip')
		X_test = pd.read_pickle('data/folds/X_test_fold_' + str(fold) + '.csv', compression='gzip')
		y_train = pd.read_pickle('data/folds/y_train_fold_' + str(fold) + '.csv', compression='gzip')
		y_test = pd.read_pickle('data/folds/y_test_fold_' + str(fold) + '.csv', compression='gzip')

		X_train = pd.merge(train, X_train, how='inner', left_index=True, right_index=True)
		X_test = pd.merge(train, X_test, how='inner', left_index=True, right_index=True)

		xgb_train = lgb.Dataset(X_train, label=y_train['DEF'], feature_name='auto')
		xgb_test = lgb.Dataset(X_test, label=y_test['DEF'], feature_name='auto')

		eval_result = {}

		lgb.train(params, xgb_train, num_boost_round=1800,
				  			valid_sets=[xgb_train, xgb_test],
				  			valid_names=['train', 'eval'],
				  			evals_result=eval_result,
                  			verbose_eval=False)

		scores[fold] = eval_result['eval']['auc']

	scores = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in scores.items()]))
	sco = scores.mean(axis=1)
	score = np.max(sco)
	cv_res = np.argmax(sco)

	logger.info('score = %f' % score)
	logger.info('best_rounds = %f' % cv_res)

	return {'loss': -sco.max(), 'status': STATUS_OK}

if __name__ == "__main__":
	logger = logging.getLogger('server_logger')
	logger.setLevel(logging.INFO)
	# create file handler which logs even debug messages
	fh = logging.FileHandler('server_lgb.log')
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

	trials = Trials()
	best = fmin(fn=score,
				space=space,
				algo=tpe.suggest,
				trials=trials,
				max_evals=150
				)



'''
2017-09-04 00:05:52 - INFO - Training with params:
2017-09-04 00:05:52 - INFO - {'num_leaves': 54, 'lambda_l1': 7.959565182343565e-05, 'learning_rate': 0.005, 'lambda_l2': 4.592386205580061e-05, 'seed': 2017, 'min_data_in_leaf': 80.0, 'bagging_fraction': 0.8834843189775895, 'min_sum_hessian_in_leaf': 9.358071448360315, 'feature_fraction': 0.9324293192906346}
2017-09-04 00:11:43 - INFO - score = 0.698426
2017-09-04 00:11:43 - INFO - best_rounds = 1474.000000
2017-09-04 00:11:43 - INFO - Training with params:
2017-09-04 00:11:43 - INFO - {'num_leaves': 47, 'lambda_l1': 6.583594775248834e-05, 'learning_rate': 0.005, 'lambda_l2': 7.248589516475117e-05, 'seed': 2017, 'min_data_in_leaf': 43.0, 'bagging_fraction': 0.8945234650798451, 'min_sum_hessian_in_leaf': 6.656461434244106, 'feature_fraction': 0.8975682143645836}
2017-09-04 00:17:01 - INFO - score = 0.698951
2017-09-04 00:17:01 - INFO - best_rounds = 1587.000000
2017-09-04 00:17:01 - INFO - Training with params:
2017-09-04 00:17:01 - INFO - {'num_leaves': 61, 'lambda_l1': 7.4691454243437e-05, 'learning_rate': 0.005, 'lambda_l2': 7.230484155948516e-05, 'seed': 2017, 'min_data_in_leaf': 55.0, 'bagging_fraction': 0.7685951513380386, 'min_sum_hessian_in_leaf': 7.323007453873621, 'feature_fraction': 0.9728211591620646}
2017-09-04 00:22:49 - INFO - score = 0.699510
2017-09-04 00:22:49 - INFO - best_rounds = 1519.000000
2017-09-04 00:22:49 - INFO - Training with params:
2017-09-04 00:22:49 - INFO - {'num_leaves': 65, 'lambda_l1': 5.309568013646748e-05, 'learning_rate': 0.005, 'lambda_l2': 9.67498799989708e-06, 'seed': 2017, 'min_data_in_leaf': 89.0, 'bagging_fraction': 0.7863694259351064, 'min_sum_hessian_in_leaf': 7.076683431242894, 'feature_fraction': 0.7689904394944868}
2017-09-04 00:28:09 - INFO - score = 0.699361
2017-09-04 00:28:09 - INFO - best_rounds = 1456.000000
'''