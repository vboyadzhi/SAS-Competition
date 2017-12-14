# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from collections import defaultdict
from sklearn.decomposition import PCA
from itertools import product
from params import col_nuls, col_less_10, RANDOM_STATE, params_xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SPLITS = 5


def woe(df_train, feature_name, target_name, df_test=None):
    def group_woe(group):
        event = float(group.sum())
        non_event = group.shape[0] - event

        rel_event = (event + 0.5) / (event_total + 0.5)
        rel_non_event = (non_event + 0.5) / (non_event_total + 0.5)

        return np.log(rel_non_event / rel_event)

    if df_train[target_name].nunique() > 2:
        raise ValueError('Target column should be binary (1/0).')

    event_total = float(df_train[df_train[target_name] == 1.0].shape[0])
    non_event_total = float(df_train.shape[0] - event_total)

    if df_test is None:
        return df_train.groupby(feature_name)[target_name].transform(group_woe)
    else:
        dict_woe = dict(zip(df_train[feature_name],
                            pd.to_numeric(df_train.groupby(feature_name)[target_name].transform(group_woe))))
        dict_woe = defaultdict(lambda: 0, dict_woe)
        return pd.to_numeric(df_train[feature_name].map(lambda x: dict_woe[x])), \
               pd.to_numeric(df_test[feature_name].map(lambda x: dict_woe[x]))


def percentile_cut(df_train, feature_name, df_test=None, qcut=100):
    if df_test is None:
        return pd.cut(df_train[feature_name].fillna(-1),
                      np.unique(
                          np.percentile(df_train[feature_name].fillna(-1), np.arange(0, 100 + 100. / qcut, 100. / qcut)
                                        )
                      )
                      , include_lowest=True)
    else:
        # union = pd.concat([df_train[feature_name].fillna(-1), df_test[feature_name].fillna(-1)], axis = 0)
        return pd.cut(df_train[feature_name].fillna(-1),
                      np.unique(
                          np.percentile(df_train[feature_name].fillna(-1), np.arange(0, 100 + 100. / qcut, 100. / qcut)
                                        )
                      )
                      , include_lowest=True).astype('object'), \
               pd.cut(df_test[feature_name].fillna(-1),
                      np.unique(
                          np.percentile(df_train[feature_name].fillna(-1), np.arange(0, 100 + 100. / qcut, 100. / qcut)
                                        )
                      )
                      , include_lowest=True).astype('object')


def mean_target(df_train, feature_name, target_name, C=None, df_test=None):
    """Mean target.
	Original idea: Stanislav Semenov
	Parameters
	----------
	C : float, default None
		Regularization coefficient. The higher, the more conservative result.
		The optimal value lies between 10 and 50 depending on the data.
	feature_name : str
	target_name : str
	df: DataFrame
	Returns
	-------
	Series
	"""

    def group_mean(group):
        group_size = float(group.shape[0])
        if C is None:
            return (group.mean() * group_size + global_mean) / group_size
        else:
            return (group.mean() * group_size + global_mean * C) / (group_size + C)

    global_mean = df_train[target_name].mean()

    if df_test is None:
        return df_train.groupby(feature_name)[target_name].transform(group_mean)
    else:
        dict_mean = dict(zip(df_train[feature_name],
                             df_train.groupby(feature_name)[target_name].transform(group_mean)))
        dict_mean = defaultdict(lambda: global_mean, dict_mean)
        return pd.to_numeric(df_train[feature_name].map(lambda x: dict_mean[x])), \
               pd.to_numeric(df_test[feature_name].map(lambda x: dict_mean[x]))


def feature_engineering(X_train, X_test, columns):

    col_model = []
    # PCA
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train[columns].fillna(0))
    X_test_sc = scaler.transform(X_test[columns].fillna(0))

    pca = PCA(n_components=25, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train_sc)
    X_train_pca = pd.DataFrame(X_train_pca, index=X_train.index)
    X_train_pca.columns = np.vectorize(lambda x: 'pca_' + str(x))(X_train_pca.columns.values)

    X_test_pca = pca.transform(X_test_sc)
    X_test_pca = pd.DataFrame(X_test_pca, index=X_test.index)
    X_test_pca.columns = np.vectorize(lambda x: 'pca_' + str(x))(X_test_pca.columns.values)
    print(pca.explained_variance_ratio_.cumsum())

    col_model.extend(X_test_pca.columns)

    # KMEAN
    clust_kmeans = KMeans(n_clusters=25, random_state=0)
    cols_for_clustering = ['sum_AMT_CREDIT_SUM', 'sum_AMT_CREDIT_SUM_OVERDUE', 'max_worst_status_60m',
                           'mean_plan_lifetime', 'min_lastclosed_time']

    X_train_sc = scaler.fit_transform(X_train[cols_for_clustering].fillna(0))
    X_test_sc = scaler.transform(X_test[cols_for_clustering].fillna(0))

    X_train_kmean = clust_kmeans.fit_transform(X_train_sc)
    X_train_kmean = pd.DataFrame(X_train_kmean, index=X_train.index)
    X_train_kmean.columns = np.vectorize(lambda x: 'kme_' + str(x))(X_train_kmean.columns.values)

    X_test_kmean = clust_kmeans.transform(X_test_sc)
    X_test_kmean = pd.DataFrame(X_test_kmean, index=X_test.index)
    X_test_kmean.columns = np.vectorize(lambda x: 'kme_' + str(x))(X_test_kmean.columns.values)


    col_model.extend(X_test_kmean.columns)

    #MEAN
    cat_features = ['min_lastcredit_time',
                    'sum_CREDIT_TYPE_AMT_CREDIT_SUM_DEBT_4',
                    'mean_CREDIT_ACTIVE_CREDIT_TYPE_AMT_CREDIT_SUM_0_5',
                    'sum_CREDIT_ACTIVE_AMT_CREDIT_SUM_0',
                    'sum_CREDIT_TYPE_AMT_CREDIT_SUM_DEBT_5',
                    'min_CREDIT_ACTIVE_lastcredit_time_1',
                    'sum_ones_0:6',
                    'sum_ones_0:3',
                    'max_worst_status_3m'
                    ]
    cont_features = ['DEF']

    col_mean = []
    for i, j in product(cat_features, cont_features):
        X_train['C_' + i], X_test['C_' + i] = \
            percentile_cut(X_train, i, qcut=10, df_test=X_test)

        X_train['MEAN_' + i + '_' + j], X_test['MEAN_' + i + '_' + j] = \
            mean_target(X_train, 'C_' + i, j, C=50, df_test=X_test)
        del X_train['C_' + i], X_test['C_' + i]
        col_mean.append('MEAN_' + i + '_' + j)


    col_model.extend(col_mean)

    return  pd.concat([X_train_pca, X_train_kmean, X_train[col_mean]], axis=1, verify_integrity=True),\
            pd.concat([X_test_pca, X_test_kmean, X_test[col_mean]], axis=1, verify_integrity=True), \
            col_model


if __name__ == "__main__":
    test = pd.concat([pd.read_pickle('data/aggregation/test.pkl', compression='gzip'),
                      pd.read_pickle('data/cross/test.pkl', compression='gzip'),
                      pd.read_pickle('data/double_cross/test.pkl', compression='gzip')], axis=1)

    target = pd.read_pickle('data/drop_duplicates/target.pkl', compression='gzip')
    train = pd.concat([pd.read_pickle('data/aggregation/train.pkl', compression='gzip'),
                       pd.read_pickle('data/cross/train.pkl', compression='gzip'),
                       pd.read_pickle('data/double_cross/train.pkl', compression='gzip'),
                       target], axis=1)

    columns_train = test.columns

    kf = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=0)
    kf.get_n_splits(train)

    for fold, (train_index, test_index) in enumerate(kf.split(np.zeros(len(train)), train['DEF'])):
        print(fold)
        # if fold > 3:
        test_folds = [fold]

        X_test = train.loc[train.index[test_index]].copy()
        X_train = train.loc[train.index[train_index]].copy()

        X_train_mean, X_test_mean, col_model = feature_engineering(X_train, X_test, columns_train)

        kf2 = StratifiedKFold(n_splits=SPLITS * 2, shuffle=True, random_state=0)
        kf2.get_n_splits(X_train)

        for train_index2, test_index2 in kf2.split(np.zeros(len(X_train)), X_train['DEF']):
            _, X_train_mean.loc[X_train.index[test_index2], :], _ = \
                feature_engineering(X_train.loc[X_train.index[train_index2], :],
                                    X_train.loc[X_train.index[test_index2], :], columns_train)

        X_train_mean[col_model].to_pickle('data/folds/X_train_fold_' + str(fold) + '.csv', compression='gzip')
        X_test_mean[col_model].to_pickle('data/folds/X_test_fold_' + str(fold) + '.csv', compression='gzip')
        X_train[['DEF']].to_pickle('data/folds/y_train_fold_' + str(fold) + '.csv', compression='gzip')
        X_test[['DEF']].to_pickle('data/folds/y_test_fold_' + str(fold) + '.csv', compression='gzip')

    _, test_mean_target, col_model = feature_engineering(train, test, columns_train)
    test_mean_target[col_model].to_pickle('data/folds/test_mean_target.csv', compression='gzip')
