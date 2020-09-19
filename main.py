################################################
## EE660 project, Prof. Jenkins, fall 2019
## Created by student Qian Wang
################################################

import numpy as np
import random
import matplotlib.pyplot as plt
import gc

from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
import math
import seaborn as sns
import plotly.express as px
import missingno as msno
from pandas.tseries.offsets import *
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib

import os
import io


class dataSet(object):
    """docstring for dataset"""

    def __init__(self, fileDir, flag):
        super(dataSet, self).__init__()
        self.fileDir = fileDir
        self.Y = None
        self.X = None
        if flag == "train":
            fileName = fileDir + "train.csv"
            rawData = pd.read_csv(fileName)
            rawData["timestamp"] = pd.to_datetime(rawData["timestamp"], format='%Y-%m-%d %H:%M:%S')
            # print("sdfsdfsdfsdfsdfsdf")
            self.rawMeter = self.compress_data(rawData)

        if flag == "test":
            fileName = fileDir + "test.csv"
            rawData = pd.read_csv(fileName)
            rawData["timestamp"] = pd.to_datetime(rawData["timestamp"], format='%Y-%m-%d %H:%M:%S')
            self.rawMeter = self.compress_data(rawData)

        fileName = fileDir + "building_metadata.csv"
        rawData = pd.read_csv(fileName)
        self.rawBuilding = self.compress_data(rawData)

        fileName = fileDir + "weather_test.csv"
        rawData = pd.read_csv(fileName)
        rawData["timestamp"] = pd.to_datetime(rawData["timestamp"], format='%Y-%m-%d %H:%M:%S')
        self.rawWeather = self.compress_data(rawData)
        self.prefilling()


        self.merged_data = self.merge_data()
        self.size = self.rawMeter.shape[0]
        # print("self.size : ",self.size)

    # filling missing data of air_temperature and dew_temperature
    def prefilling(self):
        count = 0
        length = self.rawWeather.shape[0]
        temp1 = self.rawWeather.loc[0, 'air_temperature']
        temp2 = self.rawWeather.loc[0, 'dew_temperature']
        temp3 = float(temp1)
        temp4 = float(temp2)

        for i in range(1, length):
            if np.isnan(self.rawWeather.loc[i, 'air_temperature']):
                self.rawWeather.loc[i, 'air_temperature'] = temp3
            temp1 = self.rawWeather.loc[i, 'air_temperature']
            temp3 = float(temp1)
            if np.isnan(self.rawWeather.loc[i, 'dew_temperature']):
                self.rawWeather.loc[i, 'dew_temperature'] = temp4
            temp2 = self.rawWeather.loc[i, 'dew_temperature']
            temp4 = float(temp2)



    def merge_data(self):

        self.rawMeter['timestamp'] = pd.to_datetime(self.rawMeter['timestamp'])
        temp_ds = pd.merge(self.rawMeter, self.rawBuilding, on=['building_id'], how='left')
        temp_ds = pd.merge(temp_ds, self.rawWeather, on=['site_id', 'timestamp'], how='left')

        print(temp_ds.columns.values)
        print(temp_ds.shape)
        return temp_ds

    def reset_merged_data(self):
        temp_ds = pd.merge(self.rawMeter, self.rawBuilding, on=['building_id'], how='left')
        temp_ds = pd.merge(temp_ds, self.rawWeather, on=['site_id', 'timestamp'], how='left')
        self.merged_data = temp_ds
        del temp_ds

    def compress_data(self, data):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = data.memory_usage().sum() / 1024 ** 2
        for col in data.columns:
            col_type = data[col].dtypes
            if col_type in numerics:
                c_min = data[col].min()
                c_max = data[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        data[col] = data[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        data[col] = data[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        data[col] = data[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        data[col] = data[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        data[col] = data[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        data[col] = data[col].astype(np.float32)
                    else:
                        data[col] = data[col].astype(np.float64)
        end_mem = data.memory_usage().sum() / 1024 ** 2
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,
                                                                              100 * (start_mem - end_mem) / start_mem))
        return data

    def preprocess_data(self, flag):

        self.merged_data['primary_use'] = self.merged_data['primary_use'].astype('category')
        self.merged_data['month'] = self.merged_data['timestamp'].dt.month.astype(np.int8)
        self.merged_data['week'] = self.merged_data['timestamp'].dt.weekofyear.astype(np.int8)
        self.merged_data['day_year'] = self.merged_data['timestamp'].dt.dayofyear.astype(np.int16)
        self.merged_data['day_week'] = self.merged_data['timestamp'].dt.dayofweek.astype(np.int8)
        self.merged_data['day_month'] = self.merged_data['timestamp'].dt.day.astype(np.int8)
        self.merged_data['hour_day'] = self.merged_data['timestamp'].dt.hour.astype(np.int8)
        self.merged_data['building_age'] = self.merged_data['timestamp'].dt.year - self.merged_data['year_built']
        print(self.merged_data.dtypes)
        print(type(self.merged_data.dtypes))
        print("preprocessed data : ", self.merged_data)

        self.merged_data.square_feet = np.log1p(self.merged_data.square_feet)
        # self.neighborhood()
        imputation1 = self.merged_data.groupby(['day_year', 'site_id'])['air_temperature'].mean()
        x = self.merged_data[self.merged_data['air_temperature'].isnull()][['air_temperature']]
        # print(self.merged_data.index.values)

        imputation1 = self.merged_data.groupby(['site_id'])['dew_temperature'].mean()
        x = self.merged_data[self.merged_data['air_temperature'].isnull()][['air_temperature']]
        # print(self.merged_data.index.values)
        imputation1 = self.merged_data.groupby(['site_id'])['air_temperature'].mean()
        self.merged_data.loc[self.merged_data['air_temperature'].isnull(), 'air_temperature'] = \
            self.merged_data[self.merged_data['air_temperature'].isnull()][['air_temperature']].apply(
                lambda x: imputation1[self.merged_data['site_id'][x.index]].values)
        imputation1 = self.merged_data.groupby(['site_id'])['dew_temperature'].mean()
        self.merged_data.loc[self.merged_data['dew_temperature'].isnull(), 'dew_temperature'] = \
            self.merged_data[self.merged_data['dew_temperature'].isnull()][['dew_temperature']].apply(
                lambda x: imputation1[self.merged_data['site_id'][x.index]].values)

        imputation1 = self.merged_data.groupby(['timestamp'])['wind_speed'].mean()
        self.merged_data.loc[self.merged_data['wind_speed'].isnull(), 'wind_speed'] = \
            self.merged_data[self.merged_data['wind_speed'].isnull()][['wind_speed']].apply(
                lambda x: imputation1[self.merged_data['timestamp'][x.index]].values)
        imputation1 = self.merged_data.groupby(['timestamp'])['dew_temperature'].mean()
        self.merged_data.loc[self.merged_data['wind_direction'].isnull(), 'wind_direction'] = \
            self.merged_data[self.merged_data['wind_direction'].isnull()][['wind_direction']].apply(
                lambda x: imputation1[self.merged_data['timestamp'][x.index]].values)
        imputation1 = self.merged_data.groupby(['timestamp'])['building_age'].mean()
        self.merged_data.loc[self.merged_data['building_age'].isnull(), 'building_age'] = \
            self.merged_data[self.merged_data['building_age'].isnull()][['building_age']].apply(
                lambda x: imputation1[self.merged_data['timestamp'][x.index]].values)

        imputation1 = self.merged_data.groupby(['timestamp'])['cloud_coverage'].mean()
        self.merged_data.loc[self.merged_data['cloud_coverage'].isnull(), 'cloud_coverage'] = \
            self.merged_data[self.merged_data['cloud_coverage'].isnull()][['cloud_coverage']].apply(
                lambda x: imputation1[self.merged_data['timestamp'][x.index]].values)
        imputation1 = self.merged_data.groupby(['timestamp'])['precip_depth_1_hr'].mean()
        self.merged_data.loc[self.merged_data['precip_depth_1_hr'].isnull(), 'precip_depth_1_hr'] = \
            self.merged_data[self.merged_data['precip_depth_1_hr'].isnull()][['precip_depth_1_hr']].apply(
                lambda x: imputation1[self.merged_data['timestamp'][x.index]].values)

        self.merged_data['precip_depth_1_hr'] = self.merged_data['precip_depth_1_hr'].fillna(0).astype(np.float16)
        self.merged_data['cloud_coverage'] = self.merged_data['cloud_coverage'].fillna(0).astype(np.float16)

        # drop some useless features
        drop = ["timestamp", "sea_level_pressure", "year_built", "floor_count"]
        self.merged_data.drop(drop, axis=1, inplace=True)

        le = LabelEncoder()
        self.merged_data['primary_use'] = le.fit_transform(self.merged_data['primary_use']).astype(np.int8)

        # print(self.check_missing_data())

        if flag == "standardization":

            numericals = ["primary_use", "square_feet", 'building_age', "air_temperature", "cloud_coverage",
                          "dew_temperature", 'precip_depth_1_hr', 'wind_direction', 'wind_speed', "meter", 'month',
                          'day_year', 'week', 'day_week',
                          'day_month', 'hour_day', "site_id", "building_id"]
            for str in numericals:
                mean = self.merged_data[str].mean()
                std = self.merged_data[str].std()
                standardizor = lambda x: (x - mean) / std

                self.merged_data[[str]].apply(standardizor)

        if flag == "normalization":
            categoricals = []
            numericals = ["primary_use", "square_feet", 'building_age', "air_temperature", "cloud_coverage",
                          "dew_temperature", 'precip_depth_1_hr', 'wind_direction', 'wind_speed', "meter", 'month',
                          'day_year', 'week', 'day_week',
                          'day_month', 'hour_day', "site_id", "building_id"]
            max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
            for str in numericals:
                self.merged_data[[str]].apply(max_min_scaler)

        if "meter_reading" in self.merged_data.columns:

            self.Y = np.log1p(self.merged_data.meter_reading)
            print(self.Y)
            self.X = self.merged_data.copy()
            self.X.drop("meter_reading", axis=1, inplace=True)
            print(type(self.X))
            print(self.X)

        else:
            row_ids = self.merged_data.row_id
            self.X = self.merged_data.drop("row_id", axis=1, inplace=True)

    def fill_average(self):
        average = self.merged_data.groupby(['', ''])

    # fill missing data with k nearest neighborhood
    def neighborhood(self):
        count = 0
        missingindex = self.merged_data.loc[trainSet.merged_data['air_temperature'].isnull(), ['timestamp']].copy()
        for i in missingindex.index.tolist():
            tt = self.merged_data.loc[i, 'timestamp'] + Hour()
            site = self.merged_data.loc[i, 'site_id']
            a = self.rawWeather[(self.rawWeather.timestamp == tt) & (self.rawWeather.site_id == site)].index.tolist()
            print("** i **:", i)
            print("** count **:", count)
            while not a:
                tt = tt + Hour()
                a = self.rawWeather[
                    (self.rawWeather.timestamp == tt) & (self.rawWeather.site_id == site)].index.tolist()
            self.merged_data.loc[i, 'air_temperature'] = self.rawWeather.loc[a[0], 'air_temperature']
            count += 1

        count = 0
        missingindex = self.merged_data.loc[trainSet.merged_data['dew_temperature'].isnull(), ['timestamp']].copy()
        for i in missingindex.index.tolist():
            tt = self.merged_data.loc[i, 'timestamp'] + Hour()
            site = self.merged_data.loc[i, 'site_id']
            a = self.rawWeather[
                (self.rawWeather.timestamp == tt) & (self.rawWeather.site_id == site)].index.tolist()
            print("** i **:", i)
            print("** count **:", count)
            while not a:
                tt = tt + Hour()
                a = self.rawWeather[
                    (self.rawWeather.timestamp == tt) & (self.rawWeather.site_id == site)].index.tolist()
            self.merged_data.loc[i, 'dew_temperature'] = self.rawWeather.loc[a[0], 'dew_temperature']
            count += 1

    def print_coltype(self, flag):
        type = []
        if flag == "meter":
            for col in self.rawMeter.columns:
                type.append(self.rawMeter[col].dtypes)
        elif flag == "weather":
            for col in self.rawWeather.columns:
                type.append(self.rawWeather[col].dtypes)
        elif flag == "building":
            for col in self.rawBuilding.columns:
                type.append(self.rawBuilding[col].dtypes)

        print(type)

    def check_missing_data(self):
        msno.matrix(self.merged_data.head(100000), figsize=(15, 15), width_ratios=(20, 1))
        plt.subplots_adjust(top=0.75, bottom=0.065, left=0.11, right=0.9)
        plt.savefig("missing_mergedData.png")
        plt.show()

        numMissingData = self.merged_data.isnull().sum().sort_values(ascending=False)
        percent = (self.merged_data.isnull().sum() / self.merged_data.isnull().count() * 100).sort_values(
            ascending=False)
        MissingData = pd.concat([numMissingData, percent], axis=1, keys=['Total', 'Percent'])
        return MissingData





def split_dataset(trainSet):
    categoricals = ["site_id", "building_id", "primary_use", "meter", 'month', 'day_year', 'week', 'day_week',
                    'day_month', 'hour_day']
    numericals = ["square_feet", 'building_age', "air_temperature", "cloud_coverage",
                  "dew_temperature", 'precip_depth_1_hr', 'wind_direction', 'wind_speed']

    feat_cols = categoricals + numericals
    train_X = None
    test_X = None
    train_Y = None
    test_Y = None
    kf = KFold(n_splits=3, shuffle=True, random_state=9)
    count = 0
    for train_index, test_index in kf.split(trainSet.X[feat_cols], np.array(trainSet.Y)):
        train_X = trainSet.X[feat_cols].iloc[train_index]
        test_X = trainSet.X[feat_cols].iloc[test_index]
        train_Y = trainSet.Y.iloc[train_index]
        test_Y = trainSet.Y.iloc[test_index]
        count += 1
        if count > 1:
            break
    return train_X, test_X, train_Y, test_Y


def feature_selection(trainSet):
    print("In the feature_selection function : ")

    columns_list = ['building_id', 'meter', 'site_id', 'primary_use', 'square_feet', 'air_temperature',
                    'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'wind_direction', 'wind_speed', 'month',
                    'week', 'day_year', 'day_week', 'day_month', 'hour_day', 'building_age']
    train_X, test_X, train_Y, test_Y = split_dataset(trainSet)
    print("feature before selection: ", trainSet.X.columns.values)

    a = 0.01
    clf = linear_model.Lasso(alpha=a)
    clf.fit(train_X, train_Y)

    print('Coefficients: ', clf.coef_)

    columns = []
    count = 0
    for i in clf.coef_:
        if i <= 0.000000001 and i >= -0.000000001:
            count += 1
            continue
        columns.append(columns_list[count])
        count += 1

    print("feature after selection: ", columns)
    return columns


def train_lgb(trainSet):
    categoricals = ["site_id", "building_id", "primary_use", "meter", 'month', 'day_year', 'week', 'day_week',
                    'day_month', 'hour_day']
    numericals = ["square_feet", 'building_age', "air_temperature", "cloud_coverage",
                  "dew_temperature", 'precip_depth_1_hr', 'wind_direction', 'wind_speed']

    feat_cols = categoricals + numericals
    print("int the train_lgb function: ")
    train_X = None
    test_X = None
    train_Y = None
    test_Y = None
    kf = KFold(n_splits=3, shuffle=True, random_state=9)
    count = 0
    print(trainSet.Y)
    for train_index, test_index in kf.split(trainSet.X[feat_cols], np.array(trainSet.Y)):
        train_X = trainSet.X[feat_cols].iloc[train_index]
        test_X = trainSet.X[feat_cols].iloc[test_index]
        train_Y = trainSet.Y.iloc[train_index]
        test_Y = trainSet.Y.iloc[test_index]
        count += 1
        if count > 1:
            break

    params = {
        "objective": "regression",
        "boosting": "gbdt",
        "num_leaves": 40,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "reg_lambda": 2,
        "metric": "rmse"
    }

    folds = 5
    kf = KFold(n_splits=5, shuffle=True, random_state=9)
    models = []
    for train_index, val_index in kf.split(train_X[feat_cols], np.array(train_Y)):
        train_x = train_X[feat_cols].iloc[train_index]
        val_x = train_X[feat_cols].iloc[val_index]
        train_y = train_Y.iloc[train_index]
        val_y = train_Y.iloc[val_index]
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_eval = lgb.Dataset(val_x, val_y)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=500,
                        valid_sets=(lgb_train, lgb_eval),
                        early_stopping_rounds=50,
                        verbose_eval=50)
        models.append(gbm)

    predict_Y = np.expm1(sum([model.predict(test_X[feat_cols]) for model in models]) / folds)

    error = math.sqrt(mean_squared_error(test_Y, predict_Y))
    print("first ten line of predict_Y : ", predict_Y[0:10])
    print(error)


def train_randomForest(trainSet):
    print("In the train_randomForest function : ")

    categoricals = ["site_id", "building_id", "primary_use", "meter", 'month', 'day_year', 'week', 'day_week',
                    'day_month', 'hour_day']
    numericals = ["square_feet", 'building_age', "air_temperature", "cloud_coverage",
                  "dew_temperature", 'precip_depth_1_hr', 'wind_direction', 'wind_speed']

    feat_cols = categoricals + numericals
    train_X = None
    test_X = None
    train_Y = None
    test_Y = None
    kf = KFold(n_splits=2, shuffle=True, random_state=9)
    count = 0
    # print(trainSet.Y)
    for train_index, test_index in kf.split(trainSet.X[feat_cols], np.array(trainSet.Y)):
        train_X = trainSet.X[feat_cols].iloc[train_index]
        test_X = trainSet.X[feat_cols].iloc[test_index]
        train_Y = trainSet.Y.iloc[train_index]
        test_Y = trainSet.Y.iloc[test_index]
        count += 1
        if count > 1:
            break

    temp = np.zeros((10, 6), dtype=np.float64)
    leafs = [1, 10, 20, 30, 40, 50]
    i = 0
    j = 0
    min_error = float('inf')
    best_f = 10
    best_l = 1

    for max_f in range(3, 13):
        print("i = ", i)
        for min_l in leafs:
            print("j = ", j)
            bag_x, bag_test_x, bag_y, bag_test_y = train_test_split(train_X, train_Y, test_size=0.9, random_state=666)
            clf = RandomForestRegressor(n_estimators=50, max_features=max_f, min_samples_leaf=min_l, bootstrap=True)
            clf.fit(bag_x, bag_y)
            predict_y = clf.predict(bag_test_x)
            temp[i][j] = math.sqrt(mean_squared_error(bag_test_y, predict_y))
            if min_error > temp[i][j]:
                min_error = temp[i][j]
                best_f = max_f
                best_l = min_l
            print("temp[i][j]", temp[i][j])
            j += 1

        j = 0
        i += 1

    print("best_f : ", best_f)
    print("best_l : ", best_l)
    clf = RandomForestRegressor(n_estimators=50, max_features=best_f, min_samples_leaf=best_l, bootstrap=True)
    clf.fit(train_X, train_Y)
    joblib.dump(clf, 'randomforest_model.pkl')
    predict_Y = clf.predict(test_X)
    error = math.sqrt(mean_squared_error(test_Y, predict_Y))
    print("first ten line of predict_Y : ", predict_Y[0:10])
    print("error on test dataset: ", error)


# True: train Ridge model with feature selection, False: train Ridge model without feature selection
def train_Lasso(trainSet, flag=False):
    print("In the train_Lasso function : ")
    columns = []
    train_X, test_X, train_Y, test_Y = split_dataset(trainSet)
    # print(train_X)

    if flag:
        columns = feature_selection(trainSet)
        # print('columns = ', columns)
        train_X = train_X[columns].copy()
        test_X = test_X[columns].copy()

    alpha_list = [0.001, 0.01, 0.1, 1, 10]
    # alpha_list = [0.01]
    error = np.zeros(5, dtype=np.float32)
    best_alpha = 0
    min_error = float('inf')
    count = 0
    for a in alpha_list:
        bag_x, bag_test_x, bag_y, bag_test_y = train_test_split(train_X, train_Y, test_size=0.67, random_state=666)
        clf = linear_model.Lasso(alpha=a)
        clf.fit(bag_x, bag_y)
        predict_y = clf.predict(bag_test_x)
        print('Coefficients: ', clf.coef_)

        error[count] = math.sqrt(mean_squared_error(bag_test_y, predict_y))
        if min_error > error[count]:
            min_error = error[count]
            best_alpha = a
        print(error[count])
        count += 1

    print("best_alpha : ", best_alpha)
    bag_x, bag_test_x, bag_y, bag_test_y = train_test_split(train_X, train_Y, test_size=0.5, random_state=666)
    clf = linear_model.Lasso(alpha=best_alpha)
    clf.fit(bag_x, bag_y)
    predict_Y = clf.predict(test_X)
    error = math.sqrt(mean_squared_error(test_Y, predict_Y))
    print("first ten line of predict_Y : ", predict_Y[0:10])
    print("mse = ", error)


# True: train Ridge model with feature selection, False: train Ridge model without feature selection
def train_Ridge(trainSet, flag=False):
    print("In the train_Ridge function : ")
    columns = []
    train_X, test_X, train_Y, test_Y = split_dataset(trainSet)

    if flag:
        columns = feature_selection(trainSet)
        train_X = train_X[columns].copy()
        test_X = test_X[columns].copy()

    alpha_list = [0.001, 0.01, 0.1, 1, 10]
    # alpha_list = [0.01]
    error = np.zeros(5, dtype=np.float32)
    best_alpha = 0
    min_error = float('inf')
    count = 0
    for a in alpha_list:
        bag_x, bag_test_x, bag_y, bag_test_y = train_test_split(train_X, train_Y, test_size=0.67, random_state=666)
        clf = linear_model.Ridge(alpha=a)
        clf.fit(bag_x, bag_y)
        predict_y = clf.predict(bag_test_x)
        print('Coefficients: ', clf.coef_)

        error[count] = math.sqrt(mean_squared_error(bag_test_y, predict_y))
        if min_error > error[count]:
            min_error = error[count]
            best_alpha = a
        print(error[count])
        count += 1

    print("best_alpha : ", best_alpha)
    clf = linear_model.Ridge(alpha=best_alpha)
    clf.fit(train_X, train_Y)
    # joblib.dump(clf, 'ridge_model.pkl')
    predict_Y = clf.predict(test_X)
    error = math.sqrt(mean_squared_error(test_Y, predict_Y))
    print("first ten line of predict_Y : ", predict_Y[0:10])
    print("mse = ", error)


def train_SGDRegressor(trainSet):
    print("In the train_SGDRegressor function")
    train_X, test_X, train_Y, test_Y = split_dataset(trainSet)

    error = np.zeros((5, 6), dtype=np.float64)
    iter = [1000, 2000, 3000]
    alpha_list = [0.0001, 0.001, 0.01, 0.1, 1]
    i = 0
    j = 0
    min_error = float('inf')
    best_it = 0
    best_a = 0
    for it in iter:
        print("it = ", it)
        for a in alpha_list:
            print("a = ", a)
            bag_x, bag_test_x, bag_y, bag_test_y = train_test_split(train_X, train_Y, test_size=0.9, random_state=666)
            clf = linear_model.SGDRegressor(alpha=a, max_iter=it, tol=1e-3)
            clf.fit(bag_x, bag_y)
            predict_y = clf.predict(bag_test_x)
            error[i][j] = math.sqrt(mean_squared_error(bag_test_y, predict_y))
            if min_error > error[i][j]:
                min_error = error[i][j]
                best_it = it
                best_a = a
            print("error[i][j]", error[i][j])
            j += 1

        j = 0
        i += 1

    print("best_it : ", best_it)
    print("best_a : ", best_a)
    clf = linear_model.SGDRegressor(alpha=best_a, max_iter=best_it, tol=1e-3)
    clf.fit(train_X, train_Y)
    predict_Y = clf.predict(test_X)
    error = math.sqrt(mean_squared_error(test_Y, predict_Y))
    print("first ten line of predict_Y : ", predict_Y[0:10])
    print("mse = ", error)


# True: train Ridge model with feature selection, False: train Ridge model without feature selection
def train_BayesianRidge(trainSet, flag=False):
    print("In the train_BayesianRidge function : ")
    train_X, test_X, train_Y, test_Y = split_dataset(trainSet)
    columns = []

    if flag:
        columns = feature_selection(trainSet)
        train_X = train_X[columns].copy()
        test_X = test_X[columns].copy()

    error = np.zeros((4, 4), dtype=np.float64)
    lambda_list = [0.000001, 0.0001, 0.01, 1]
    alpha_list = [0.000001, 0.0001, 0.01, 1]
    i = 0
    j = 0
    min_error = float('inf')
    best_lam = 0
    best_a = 0
    for lam in lambda_list:
        print("it = ", lam)
        for a in alpha_list:
            print("a = ", a)
            bag_x, bag_test_x, bag_y, bag_test_y = train_test_split(train_X, train_Y, test_size=0.9, random_state=666)
            clf = linear_model.BayesianRidge(alpha_2=a, lambda_2=lam)
            clf.fit(bag_x, bag_y)
            predict_y = clf.predict(bag_test_x)
            error[i][j] = math.sqrt(mean_squared_error(bag_test_y, predict_y))
            if min_error > error[i][j]:
                min_error = error[i][j]
                best_lam = lam
                best_a = a
            print("error[i][j]", error[i][j])
            j += 1

        j = 0
        i += 1

    print("best_it : ", best_lam)
    print("best_a : ", best_a)
    clf = linear_model.BayesianRidge(alpha_2=best_a, lambda_2=best_lam)
    clf.fit(train_X, train_Y)
    predict_Y = clf.predict(test_X)
    error = math.sqrt(mean_squared_error(test_Y, predict_Y))
    print("first ten line of predict_Y : ", predict_Y[0:10])
    print("mse = ", error)


def train_SVR(trainSet, flag=False):
    print("In the train_Ridge function : ")
    columns = []
    train_X, test_X, train_Y, test_Y = split_dataset(trainSet)

    if flag:
        columns = feature_selection(trainSet)
        train_X = train_X[columns].copy()
        test_X = test_X[columns].copy()

    C_list = [0.001, 0.01, 0.1, 1, 10]
    error = np.zeros(5, dtype=np.float32)
    best_C = 0
    min_error = float('inf')
    count = 0
    for c in C_list:
        print("c = ", c)
        bag_x, bag_test_x, bag_y, bag_test_y = train_test_split(train_X, train_Y, test_size=0.9, random_state=666)
        clf = SVR(C=c, gamma='scale')
        clf.fit(bag_x, bag_y)
        predict_y = clf.predict(bag_test_x)

        error[count] = math.sqrt(mean_squared_error(bag_test_y, predict_y))
        if min_error > error[count]:
            min_error = error[count]
            best_C = c
        print(error[count])
        count += 1

    print("best_C : ", best_C)
    clf = SVR(C=best_C, gamma='scale')
    clf.fit(train_X, train_Y)
    predict_Y = clf.predict(test_X)
    error = math.sqrt(mean_squared_error(test_Y, predict_Y))
    print("first ten line of predict_Y : ", predict_Y[0:10])
    print("mse = ", error)


def train_KNeighbors(trainSet, flag=False):
    print("In the train_Ridge function : ")
    columns = []
    train_X, test_X, train_Y, test_Y = split_dataset(trainSet)

    if flag:
        columns = feature_selection(trainSet)
        train_X = train_X[columns].copy()
        test_X = test_X[columns].copy()

    n_list = [2, 5, 10, 20]
    error = np.zeros(4, dtype=np.float32)
    best_N = 0
    min_error = float('inf')
    count = 0
    for n in n_list:
        print("n = ", n)
        bag_x, bag_test_x, bag_y, bag_test_y = train_test_split(train_X, train_Y, test_size=0.85, random_state=666)
        clf = KNeighborsRegressor(n_neighbors=n)
        clf.fit(bag_x, bag_y)
        predict_y = clf.predict(bag_test_x)

        error[count] = math.sqrt(mean_squared_error(bag_test_y, predict_y))
        if min_error > error[count]:
            min_error = error[count]
            best_N = n
        print(error[count])
        count += 1

    print("best_N : ", best_N)
    clf = KNeighborsRegressor(n_neighbors=best_N)
    clf.fit(train_X, train_Y)
    predict_Y = clf.predict(test_X)
    error = math.sqrt(mean_squared_error(test_Y, predict_Y))
    print("first ten line of predict_Y : ", predict_Y[0:10])
    print("mse = ", error)

def run_randomForest(trainSet):
    print("In the train_randomForest function : ")

    categoricals = ["site_id", "building_id", "primary_use", "meter", 'month', 'day_year', 'week', 'day_week',
                    'day_month', 'hour_day']
    numericals = ["square_feet", 'building_age', "air_temperature", "cloud_coverage",
                  "dew_temperature", 'precip_depth_1_hr', 'wind_direction', 'wind_speed']

    feat_cols = categoricals + numericals
    train_X = None
    test_X = None
    train_Y = None
    test_Y = None
    kf = KFold(n_splits=2, shuffle=True, random_state=9)
    count = 0
    # print(trainSet.Y)
    for train_index, test_index in kf.split(trainSet.X[feat_cols], np.array(trainSet.Y)):
        train_X = trainSet.X[feat_cols].iloc[train_index]
        test_X = trainSet.X[feat_cols].iloc[test_index]
        train_Y = trainSet.Y.iloc[train_index]
        test_Y = trainSet.Y.iloc[test_index]
        count += 1
        if count > 1:
            break



    clf = joblib.load('randomforest_model.pkl')
    predict_Y = clf.predict(test_X)
    error = math.sqrt(mean_squared_error(test_Y, predict_Y))
    print("first ten line of predict_Y : ", predict_Y[0:10])
    print("error on test dataset: ", error)

if __name__ == "__main__":
    Root = os.path.abspath(os.curdir)
    fileDir = Root + "/data/"

    trainSet = dataSet(fileDir, "test")
    # trainSet.print_coltype("meter")
    # trainSet.print_coltype("weather")
    # trainSet.print_coltype("building")



    trainSet.preprocess_data("normalization")

    # missingData = trainSet.check_missing_data()
    # print(missingData)
    # print(trainSet.merged_data.loc[trainSet.merged_data['air_temperature'].isnull(),['timestamp','site_id']])
    # trainSet.merged_data.loc[trainSet.merged_data['air_temperature'].isnull(), ['timestamp', 'site_id']].to_csv('out.csv')

    # trainSet.merged_data['day_year'].to_csv('out1.csv')

    run_randomForest(trainSet)


