import gzip
import os
from io import BytesIO
from ish_parser import ish_parser

def read_observations(years, usaf='081810', wban='99999'):
    parser = ish_parser()
    
    for year in years:
        path = "../data/observations/{usaf}-{wban}-{year}.gz".format(year=year, usaf=usaf, wban=wban)
        with gzip.open(path) as gz:
            parser.loads(bytes.decode(gz.read()))
            
    reports = parser.get_reports()
    
    station_latitudes = [41.283, 41.293]
    # station_latitudes = [40.080, 40.090]
    observations = pd.DataFrame.from_records(((r.datetime, 
                                               r.air_temperature.get_numeric(),
                                               (r.precipitation[0]['depth'].get_numeric() if r.precipitation else 0),
                                               r.humidity.get_numeric(),
                                               r.sea_level_pressure.get_numeric(),
                                               r.wind_speed.get_numeric(),
                                               r.wind_direction.get_numeric()) 
                                              for r in reports if r.latitude in station_latitudes and r.datetime.minute == 0),
                             columns=['timestamp', 'AT', 'precipitation', 'humidity', 'pressure', 'wind_speed', 'wind_direction'], 
                             # columns=['timestamp', 'AT', 'precipitation', 'humidity', 'wind_speed', 'wind_direction'],
                             index='timestamp')
    
    return observations


import json
import pandas as pd
import numpy as np

years = range(2013, 2018)
dataset = read_observations(years)

original = dataset.copy(deep=True)



from sklearn import preprocessing

pd.options.mode.chained_assignment = None
np.random.seed(1234)

def drop_duplicates(df):
    print("Number of duplicates: {}".format(len(df.index.get_duplicates())))
    return df[~df.index.duplicated(keep='first')]
    
def impute_missing(df):
    # todo test with moving average / mean or something smarter than forward fill
    print("Number of rows with nan: {}".format(np.count_nonzero(df.isnull())))
    df.fillna(method='ffill', inplace=True)
    return df
    
def first_order_difference(data, columns):
    for column in columns:
        data[column+'_d'] = data[column].diff(periods=1)
    
    return data.dropna()

# def derive_prediction_columns(data, column, column2, horizons):
#     for look_ahead in horizons:
#         data['prediction_' + str(look_ahead)] = data[column].diff(periods=look_ahead).shift(-look_ahead)
        
#     for look_ahead in horizons:
#         data['prediction_direction_' + str(look_ahead)] = data[column2].diff(periods=look_ahead).shift(-look_ahead)

def derive_prediction_columns(data, column, horizons):
    for look_ahead in horizons:
        data['prediction_' + str(look_ahead)] = data[column].diff(periods=look_ahead).shift(-look_ahead)
    
    return data.dropna()

def scale_features(scaler, features):
    # scaler.fit(features)
    
    scaled = scaler.fit_transform(features)
    scaled = pd.DataFrame(scaled, columns=features.columns)
    
    return scaled

def inverse_prediction_scale(scaler, predictions, original_columns, column):
    loc = original_columns.get_loc(column)
    
    inverted = np.zeros((len(predictions), len(original_columns)))
    inverted[:,loc] = np.reshape(predictions, (predictions.shape[0],))
    
    inverted = scaler.inverse_transform(inverted)[:,loc] # Scale back the data to the original representation
    
    return inverted

# def invert_all_prediction_scaled(scaler, predictions, original_columns, horizons):
#     inverted = np.zeros(predictions.shape)
#     inverted2 = np.zeros(predictions.shape)
    
#     for col_idx, horizon in enumerate(horizons):
#         inverted[:,col_idx] = inverse_prediction_scale(
#             scaler, predictions[:,col_idx], 
#             original_columns,
#             "prediction_" + str(horizon))

#     for col_idx, horizon in enumerate(horizons):
#         inverted2[:,col_idx] = inverse_prediction_scale(
#             scaler, predictions[:,col_idx], 
#             original_columns,
#             "prediction_direction_" + str(horizon))
    
#     return inverted, inverted2

def invert_all_prediction_scaled(scaler, predictions, original_columns, horizons):
    inverted = np.zeros(predictions.shape)
    
    for col_idx, horizon in enumerate(horizons):
        inverted[:,col_idx] = inverse_prediction_scale(
            scaler, predictions[:,col_idx], 
            original_columns,
            "prediction_" + str(horizon))
    
    return inverted

def inverse_prediction_difference(predictions, original):
    return predictions + original

def invert_all_prediction_differences(predictions, original):
    inverted = predictions
    
    for col_idx, horizon in enumerate(horizons):
        inverted[:, col_idx] = inverse_prediction_difference(predictions[:,col_idx], original)
        
    return inverted


dataset = drop_duplicates(dataset)
dataset = impute_missing(dataset)
dataset.describe()

#select features we're going to use
# features = dataset[['wind_speed', 
#                     'nems4_wind_speed',
#                     'wind_direction',
#                     'nems4_wind_direction', 
#                     'AT', 
#                     'nems4_AT', 
#                     'humidity', 
#                     'nems4_humidity',
#                     'pressure',
#                     'nems4_pressure']]

features = dataset[['wind_speed', 
                    'wind_direction', 
                    'AT', 
                    'precipitation',
                    'humidity',
                    'pressure'
                    ]]

# the time horizons we're going to predict (in hours)
horizons = [1, 2, 3, 4]

features = first_order_difference(features, features.columns)
features = derive_prediction_columns(features, 'wind_speed', horizons)
# features = derive_prediction_columns(features, 'wind_speed', 'wind_direction', horizons)

scaler = preprocessing.StandardScaler()
# scaler = preprocessing.MinMaxScaler()
scaled = scale_features(scaler, features)

scaled.describe()


# In[ ]:

def prepare_test_train(data, features, predictions, sequence_length, split_percent=0.9):
    
    num_features = len(features)
    num_predictions = len(predictions)
    
    # make sure prediction cols are at end
    columns = features + predictions
    
    data = data[columns].values
    
    print("Using {} features to predict {} horizons".format(num_features, num_predictions))
    
    result = []
    for index in range(len(data) - sequence_length+1):
        result.append(data[index:index + sequence_length])

    result = np.array(result)
    # shape (n_samples, sequence_length, num_features + num_predictions)
    print("Shape of data: {}".format(np.shape(result)))
    
    row = round(split_percent * result.shape[0])
    train = result[:row, :]
    
    X_train = train[:, :, :-num_predictions]
    y_train = train[:, -1, -num_predictions:]
    X_test = result[row:, :, :-num_predictions]
    y_test = result[row:, -1, -num_predictions:]
    
    print("Shape of X train: {}".format(np.shape(X_train)))
    print("Shape of y train: {}".format(np.shape(y_train)))
    print("Shape of X test: {}".format(np.shape(X_test)))
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_features))
    
    y_train = np.reshape(y_train, (y_train.shape[0], num_predictions))
    y_test = np.reshape(y_test, (y_test.shape[0], num_predictions))
    
    return X_train, y_train, X_test, y_test, row


# In[ ]:

sequence_length = 48

prediction_cols = ['prediction_' + str(h) for h in horizons]
# prediction_cols_2 = ['prediction_direction_' + str(h) for h in horizons]
# prediction_cols = prediction_cols + prediction_cols_2

# 多特征
# feature_cols = ['wind_speed_d', 'nems4_wind_speed_d', 
#                 # 'wind_direction_d', 'nems4_wind_direction_d',
#                 'AT_d', 'nems4_AT_d', 
#                 'humidity_d', 'nems4_humidity_d', 
#                 'pressure_d', 'nems4_pressure_d']

# 少特征
feature_cols = ['wind_speed_d', 
#                 'wind_direction_d', 
                'AT_d',
                # 'precipitation_d', 
                'humidity_d',
                'pressure_d'
                ]

# 不做一阶差分的特征项
# feature_cols = ['wind_speed', 
# #                 'wind_direction', 
#                 'AT', 
#                 'humidity', 
#                 'pressure']

X_train, y_train, X_test, y_test, row_split = prepare_test_train(
    scaled,
    feature_cols,
    prediction_cols,
    sequence_length,
    split_percent = 0.9)


# In[ ]:

from sklearn.metrics import mean_squared_error, mean_absolute_error

#(-1 is because we only take the last y row in each sequence)
sequence_offset = sequence_length - 1

# validate train
inverse_scale = invert_all_prediction_scaled(scaler, y_train, scaled.columns, horizons)

assert(mean_squared_error(
    features[prediction_cols][sequence_offset:row_split+sequence_offset], 
    inverse_scale) < 1e-10)


undiff_prediction = invert_all_prediction_differences(
    inverse_scale, 
    features['wind_speed'][sequence_offset:row_split+sequence_offset])

for i, horizon in enumerate(horizons):
    assert(mean_squared_error(
        features['wind_speed'][sequence_offset+horizon:row_split+sequence_offset+horizon], 
        undiff_prediction[:,i]) < 1e-10)

    
# validate test
inverse_scale = invert_all_prediction_scaled(scaler, y_test, scaled.columns, horizons)

assert(mean_squared_error(
    features[prediction_cols][sequence_offset+row_split:], 
    inverse_scale) < 1e-10)

undiff_prediction = invert_all_prediction_differences(
    inverse_scale, 
    features['wind_speed'][sequence_offset+row_split:])

for i, horizon in enumerate(horizons):
    assert(mean_squared_error(
        features['wind_speed'][sequence_offset+row_split+horizon:], 
        undiff_prediction[:-horizon,i]) < 1e-10)


filename = os.path.basename(__file__)
batch_size = 1024
epochs = 300

# Build the LSTM Model

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.optimizers import RMSprop, Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.grid_search import GridSearchCV
from keras.utils import plot_model
from keras import losses


def build_model(layers):
    model = Sequential()

    model.add(LSTM(64, 
            dropout=0.2, 
            recurrent_dropout=0.1, 
            input_shape=(sequence_length, layers[0]), 
            return_sequences=True))
    # model.add(Dropout(0.2))
    
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(16))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='linear'))
    model.add(Dense(64, activation='linear'))

    model.add(Dense(layers[1]))
    model.add(Activation('linear'))
    
    model.compile(loss='mae', optimizer='rmsprop')
    
    print(model.summary())
    plot_model(model, to_file='./model/model_'+filename+'.png', show_shapes=True)
          
    return model


def run_network(X_train, y_train, X_test, y_test, layers, epochs, batch_size=batch_size):
    model = build_model(layers)
    history = None
    
    try:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size, 
            epochs=epochs,
            validation_split=0.1,
            callbacks=[
                TensorBoard(log_dir='./logs', write_graph=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=60, 
                                verbose=1, mode='auto', min_lr=0.0001),
                EarlyStopping(monitor='val_loss', patience=80, verbose=1, mode='auto'),
                ModelCheckpoint('./model/best_'+filename+'.hdf5', monitor='val_loss', verbose=1, 
                                save_best_only=True, mode='auto')
            ])
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    
    predicted = model.predict(X_test)
    scores = model.evaluate(X_test, y_test, verbose=0)
    
    return model, predicted, history, scores


model, predicted, history, scores = run_network(
    X_train, 
    y_train, 
    X_test,
    y_test,
    layers = [X_train.shape[2], y_train.shape[1]],
    epochs=epochs)


print(scores)
print(history.history.keys())


filename = os.path.basename(__file__)

import matplotlib.pyplot as plt
# loss
fig = plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.xlabel('epoch')
plt.savefig('./image/loss_' + filename + '.png')
plt.show()


# print("*********************************************************")
# print(type(predicted)) # <class 'numpy.ndarray'>
# print(predicted.shape) # (7594, 6)
print("*********************************************************")
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

print("MAE {:.3}, RMSE {:.3f}".format(
    mean_absolute_error(y_test, predicted),
    sqrt(mean_squared_error(y_test, predicted))))

# for i, horizon in enumerate(horizons):
#     print("wind speed: MAE {:.3f}, RMSE {:.3f} for horizon {}".format(
#         mean_absolute_error(y_test[:,i], predicted[:,i]),
#         sqrt(mean_squared_error(y_test[:,i], predicted[:,i])),
#         horizon))

# for i, horizon in enumerate(horizons):
#     print("wind direction: MAE {:.3f}, RMSE {:.3f} for horizon {}".format(
#         mean_absolute_error(y_test[:,i], predicted[:,i]),
#         sqrt(mean_squared_error(y_test[:,i], predicted[:,i])),
#         horizon))


# In[ ]:

sequence_offset = sequence_length - 1

# inverse_scale, inverse_scale2 = invert_all_prediction_scaled(scaler, predicted, scaled.columns, horizons)
inverse_scale = invert_all_prediction_scaled(scaler, predicted, scaled.columns, horizons)
# print(inverse_scale.shape) # (7594, 6)
# print("*********************************************************************")
# print(inverse_scale2.shape) # (7594, 6)

predicted_signal = invert_all_prediction_differences(
    inverse_scale, 
    features['wind_speed'][sequence_offset+row_split:])

# print(predicted_signal[:5])
# print("Shape of predicted_signal:", predicted_signal.shape) # (2588, 3)

# predicted_signal2 = invert_all_prediction_differences(
#     inverse_scale2, 
#     features['wind_direction'][sequence_offset+row_split:])
# print(predicted_signal2[:5])
# print("***************************************************************************")

for i, horizon in enumerate(horizons):
    a = features['wind_speed'][sequence_offset+row_split+horizon:]
    p = predicted_signal[:-horizon, i]
    # print("Shape of a:", a.shape)  # (7591,) (7590,) (7589,) (7588,)
    # print("Shape of p:", p.shape)  # (7591,) (7590,) (7589,) (7588,)
    mae = mean_absolute_error(a, p)
    rmse = sqrt(mean_squared_error(a, p))
    mape = np.average(np.abs((a-p)/p))*100
    nrmse_mean = 100*rmse/(a.mean())
    
    print("Real scale wind speed: MAE {:.3f}, RMSE {:.3f}, MAPE {:.3f}, NRMSE {:.3f} for horizon {}".format(
            mae, rmse, mape, nrmse_mean, horizon))
    
# for i, horizon in enumerate(horizons):
#     a2 = features['wind_direction'][sequence_offset+row_split+horizon:]
#     p2 = predicted_signal2[:-horizon, i]

#     print("Real scale wind direction: MAE {:.3f}, RMSE {:.3f} for horizon {}".format(
#             mean_absolute_error(a2, p2),
#             sqrt(mean_squared_error(a2, p2)),
#             horizon))


plot_samples=500
max_horizon = horizons[-1]
plots = len(horizons)

fig = plt.figure(figsize=(14, 5 * plots))
fig.suptitle("Model Prediction at each Horizon")

for i, horizon in enumerate(horizons):
    plt.subplot(plots, 1, i+1)
    
    len_adjust = max_horizon-horizon # ensure all have same lenght
    
    real = features['wind_speed'][sequence_offset+row_split+horizon+len_adjust:].values
    pred = predicted_signal[len_adjust:-horizon,i]
    
    plt.plot(real[:plot_samples], label='observed')
    plt.plot(pred[:plot_samples], label='predicted')
    plt.title("Prediction for {} Hour Horizon".format(horizon))
    plt.xlabel("Hour")
    plt.ylabel("Wind Speed (m/s)")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('./image/result_' + filename + '.png')
    
fig.tight_layout()
plt.subplots_adjust(top=0.95)  