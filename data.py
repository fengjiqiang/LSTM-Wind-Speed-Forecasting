import gzip
from io import BytesIO
from ish_parser import ish_parser
import pandas as pd 
import numpy as np 

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

years = range(2013, 2018)
dataset = read_observations(years)
print(dataset.shape)
print(dataset[:20])
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


dataset = drop_duplicates(dataset)
dataset = impute_missing(dataset)

features = dataset[['wind_speed', 
                    'wind_direction', 
                    'AT', 
                    'humidity', 
                    'pressure']]

# print(features.shape)
print("***********************************************************")
print(dataset[:20])