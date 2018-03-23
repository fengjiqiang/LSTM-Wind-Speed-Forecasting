import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_color_codes()

import gzip
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
    observations = pd.DataFrame.from_records(((r.datetime, 
                                               r.air_temperature.get_numeric(),
                                               (r.precipitation[0]['depth'].get_numeric() if r.precipitation else 0),
                                               r.humidity.get_numeric(),
                                               r.sea_level_pressure.get_numeric(),
                                               r.wind_speed.get_numeric(),
                                               r.wind_direction.get_numeric()) 
                                              for r in reports if r.latitude in station_latitudes and r.datetime.minute == 0),
                             columns=['timestamp', 'AT', 'precipitation', 'humidity', 'pressure', 'wind_speed', 'wind_direction'], 
                             index='timestamp')
    
    return observations


nems4_lookahead = 24

def read_nems4(years, prediction_hours=12):
    predictions=pd.DataFrame()
    for year in years:
        with open('data/NEMS4/{}.json'.format(year)) as json_data:
            d = json.load(json_data)
            if not predictions.empty:
                predictions = predictions.append(pd.DataFrame(d['history_1h']))
            else:
                predictions = pd.DataFrame(d['history_1h'])

    predictions = predictions.set_index('time')
    predictions.index.name = 'timestamp'
    
    # shift dataset back 12 hours as its a the value is the prediction for the given timestmap 12 hours previously
    predictions.index = pd.to_datetime(predictions.index) - pd.Timedelta(hours=nems4_lookahead)
    predictions.index.tz = 'UTC'

    predictions = predictions[['temperature', 'precipitation', 
                   'relativehumidity', 'sealevelpressure', 
                   'windspeed', 'winddirection']]
    
    predictions = predictions.rename(columns={
        'windspeed': 'nems4_wind_speed',
        'winddirection': 'nems4_wind_direction', 
        'temperature': 'nems4_AT',
        'precipitation': 'nems4_precipitation',
        'relativehumidity': 'nems4_humidity',
        'sealevelpressure': 'nems4_pressure'})
    
    return predictions


years = range(2013, 2018)
# dataset = pd.merge(read_observations(years), read_nems4(years), left_index=True, right_index=True, how='inner')
dataset = read_observations(years)

original = dataset.copy(deep=True)
dataset.describe()


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


dataset = drop_duplicates(dataset)
dataset = impute_missing(dataset)

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
                    'humidity', 
                    'pressure']]


import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from scipy import signal
import peakutils as peak 

row = round(0.9 * features.shape[0])
# print(type(features))  # <class 'pandas.core.frame.DataFrame'>
train = features['wind_speed'][:row]
print(train.head())
test = features['wind_speed'][row:]

def test_stationarity(timeseries, window=12):
    df = pd.DataFrame(timeseries)
    df['Rolling.Mean'] = timeseries.rolling(window=window).mean()
    df['Rolling.Std']=timeseries.rolling(window=window).std()
    adftest = sm.tsa.stattools.adfuller(timeseries)
    adfoutput = pd.Series(adftest[0:4], index=['统计量','p-值','滞后量','观测值数量'])
    for key,value in adftest[4].items():
        adfoutput['临界值 (%s)'% key] = value
    return adfoutput, df
    
fig = plt.figure(figsize=(12, 5))
adftest, dftest0 = test_stationarity(train)
plt.plot(dftest0)
plt.legend(["raw data", "roll mean", "roll std"])
# plt.show()
print('原始数据平稳性检验:')
print(adftest)

# identify cycles
def CalculateCycle(ts, lags=36):
    acf_x, acf_ci = acf(ts, alpha=0.05, nlags=lags)
    fs=1
    f, Pxx_den = signal.periodogram(acf_x, fs)
    
    index = peak.indexes(Pxx_den)
    cycle=(1/f[index[0]]).astype(int)
    fig = plt.figure()
    ax0 = fig.add_subplot(111)
    plt.vlines(f, 0, Pxx_den)
    plt.plot(f, Pxx_den, marker='o', linestyle='none', color='red')
    plt.title("Identified Cycle of %i" % (cycle))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    # plt.show()
    # return(index, f, Pxx_den)

CalculateCycle(train)

# test stationarity after removing seasonality
Seasonality = 8760
windSpeed = train.diff(Seasonality)[Seasonality:]
adftestS12 = sm.tsa.stattools.adfuller(windSpeed)
print("ADF test result shows test statistic is %f and p-value is %f" %(adftestS12[:2]))

nlag = 48
xvalues = np.arange(nlag+1)

acfS12, confiS12 = sm.tsa.stattools.acf(windSpeed, nlags=nlag, alpha=0.05, fft=False)
confiS12 = confiS12 - confiS12.mean(1)[:,None]

fig = plt.figure()
ax0 = fig.add_subplot(221)
windSpeed.plot(ax=ax0)

ax1=fig.add_subplot(222)
sm.graphics.tsa.plot_acf(windSpeed, lags=nlag, ax=ax1)
# plt.show()

fig = plt.figure()
ax0 = fig.add_subplot(221)
sm.graphics.tsa.plot_acf(windSpeed, ax=ax0, lags=48)

ax1 = fig.add_subplot(222)
sm.graphics.tsa.plot_pacf(windSpeed, ax=ax1, lags=48)
# plt.show()

# Build SARIMA
model = sm.tsa.statespace.SARIMAX(train, trend='n', order=(0,1,0), seasonal_order=(0,1,1,8760)).fit()
pred = model.predict()
print(model.summary())


# check validation results
from sklearn.metrics import mean_squared_error, mean_absolute_error

MAPE = np.average(np.abs((train-pred)/train))*100
MAE = mean_absolute_error(train, pred)


fig = plt.figure()
ax0 = fig.add_subplot(211)
plt.plot(train, label='Original')
plt.plot(pred, label='Fitted')
plt.legend(loc='best')
plt.title("SARIMA(0,1,0)(0,1,1,18) Model, MAE {:.3f}".format(MAE))
plt.savefig('./image/sarima.png')

# ax1 = fig.add_subplot(212)
# plt.plot(subtrain, color='red', label='Original')
# plt.plot(subpred, label='Fitted')
# plt.legend(loc='best')
# plt.title("SARIMA(0,1,0)(0,1,1,18) Model, MAE {:.3f}".format(subMAE))
# plt.savefig('./image/sarima.png')

# test alternatives
# model2 = sm.tsa.statespace.SARIMAX(train, trend='n', order=(1,1,1), seasonal_order=(0,1,1,18)).fit()
# forecast1 = model.predict(start='2017-12-31 00:00:00+00:00', end='2017-12-31 01:00:00+00:00', dynamic=True)
# # forecast2 = model2.predict(start='1976-12-01 00:00:00', end='1978', dynamic=True)
# MAPE1 = ((test-forecast1).abs()/test).mean()*100
# # MAPE2 = ((test-forecast2).abs()/test).mean()*100

# plt.plot(test, color='black', label='Original')
# plt.plot(forecast1, color='green', label='Model 1 : SARIMA(0,1,0)(0,1,1,18)')
# # plt.plot(forecast2, color='red', label='Model 2 : SARIMA(1,1,1)(0,1,1,18)')
# plt.legend(loc='best')
# plt.title('Model 1 MAPE=%.f%%; Model 2 MAPE=%.f%%'%(MAPE1, MAPE2))
plt.show()