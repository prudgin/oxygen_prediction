import logging
from datetime import datetime
from typing import Dict
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import helper_functions as hlp

"""
Dataset Description:
The dataset contains information about the water quality and weather conditions of aquaculture ponds.
The data was collected with the aim of predicting and analyzing the levels of dissolved oxygen in the pond water.

The dataset consists of the following columns:

- date_time: The date and time when the measurement was taken.
- pond_no: The identifier of the pond where the measurement was taken.
    {'np1': 1, 'np2': 2, 'vp1': 3, 'vp2': 4, 'vp3': 5, 'vp4': 6}
- date: The date of the measurement.
- time: The time of the measurement.
- meas_sec: The seconds passed since the day start (00:00:00) to the moment the water measurement was taken.
- is_eve: A boolean value indicating whether the measurement was taken in the evening.
- meas_lag: The lag in seconds between the measurement and sunrise/sunset.
- o2: The level of dissolved oxygen in the pond water.
- temp: The temperature of the water in the pond.
- pressure: The atmospheric pressure recorded on the day of the measurement, in millimeters of mercury (mmHg).
- o2_sat: The level of oxygen saturation in the water as a part of max saturation at the given temperature and pressure.
- secchi: The transparency of the water, measured in meters by Secchi disc visibility.
- feed_kg: the amount of fish feed given to a pond.
- sun_elev: sinus of max sun elevation angle for the given date.
- moon_phase: The phase of the moon.
- max_temp: The maximum air temperature recorded on the day of the measurement.
- min_temp: The minimum air temperature recorded on the day of the measurement.
- avg_temp: The average air temperature recorded on the day of the measurement.
- sun_hr: The duration of sunlight in hours (maybe? it takes into accound cloud coverage somehow).
- uv_idx: The UV index recorded on the day of the measurement.
- wind_spd: The wind speed recorded on the day of the measurement, in kilometers per hour.
- wind_dir: The wind direction recorded on the day of the measurement, in degrees.
- weath_code: A code representing the overall weather conditions on the day of the measurement.
              codes can be found here: https://www.worldweatheronline.com/weather-api/api/docs/weather-icons.aspx
- precip_mm: The amount of precipitation recorded on the day of the measurement, in millimeters.
- humidity: The humidity level recorded on the day of the measurement.
- cloud_cov: The percentage of the sky covered by clouds on the day of the measurement.
- dew_pt: The dew point temperature recorded on the day of the measurement, in degrees Celsius.
- wind_gust: The gust wind speed recorded on the day of the measurement, in kilometers per hour.
- moonshine: The duration of moonshine multiplied by the moon illuminated surface and by max moon elevation (sin)
- sun_sec: The duration of sunshine, in seconds.
- cloud_desc: A description of the cloud conditions on the day of the measurement.

The water measurements were taken at dusk or dawn.
The weather data is provided for every 24-hour interval from www.worldweatheronline.com

In this file the described dataset is split by pond and all NaNs are filled
"""

logging.basicConfig(level=logging.INFO)


class PondDataHolder:
    """
    This class takes a file with weather and pond data and transforms it
    into a dict of pond dataframes.
    """

    def __init__(
            self,
            ponds_encode: Dict,  # like {'np1': 1, ... }
            filename='pond_weather_combined_23.07.2023.csv',  # name of the file with raw data
    ):
        self.filename = filename
        self.ponds: Dict[str, pd.DataFrame] = {}
        self.min_date = datetime.today().date()
        self.ponds_encode = ponds_encode
        self.pond_names = []
        self.na_shifted_seasonability_rows: Dict[str, pd.DataFrame] = {}

    def populate_ponds_dict(self,
                            ponds_to_drop: list  # a list of pond names to drop
                            ):
        df = pd.read_csv(self.filename)

        self.pond_names = list(df['pond_no'].unique())
        for rm in ponds_to_drop:
            self.pond_names.remove(rm)

        df['date'] = pd.to_datetime(df['date']).dt.date
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.sort_values(by=['date_time', 'pond_no'])

        for pond in self.pond_names:
            pond_df = df.loc[df['pond_no'] == pond]

            # Encode pond name
            pond_df.loc[:, 'pond_no'] = self.ponds_encode[pond]

            # Keep only data between valid indexes (not NaN values)
            pond_df = pond_df.loc[
                      pond_df['o2'].first_valid_index()
                      :pond_df['o2'].last_valid_index()
                      ]
            pond_df.sort_values(by=['date_time'], inplace=True)
            pond_df = pond_df.set_index('date_time')

            for sun in ['sunrise', 'sunset']:
                pond_df[sun] = pd.to_datetime(pond_df['date'].astype(str) + ' ' + pond_df[sun].astype(str))

            # Fill nans

            # evening and morning data interpolation
            for is_eve in [True, False]:
                mask = pond_df['is_eve'] == is_eve
                pond_df.loc[mask, 'o2'] = pond_df.loc[mask, 'o2'].interpolate(method='akima')
                pond_df.loc[mask, 'temp'] = pond_df.loc[mask, 'temp'].interpolate(method='akima')

            # DO saturation. Recalculate after interpolation
            pond_df['o2_sat'] = pond_df['o2'] / pond_df.apply(
                lambda row: hlp.oxygen_saturation(row['temp'], row['pressure']),
                axis=1
            )
            # Secchi visibility fill nans
            pond_df['secchi'] = pond_df['secchi'].interpolate(method='akima')
            pond_df['secchi'] = pond_df['secchi'].fillna(-1)

            # Encode the categorical variables
            le = LabelEncoder()
            pond_df['cloud_desc'] = le.fit_transform(pond_df['cloud_desc'])
            pond_df['moon_phase'] = le.fit_transform(pond_df['moon_phase'])

            # Calculate air temp_delta from min and max
            pond_df['delta_temp'] = pond_df['max_temp'] - pond_df['min_temp']

            self.min_date = min(self.min_date, pond_df['date'].min())

            self.ponds[pond] = pond_df

        # Set column 'days_passed'
        for pond in self.pond_names:
            self.ponds[pond]['date'] = pd.to_datetime(self.ponds[pond]['date'])
            self.ponds[pond]['days_passed'] = (self.ponds[pond]['date'] - pd.Timestamp(self.min_date)).dt.days


    def eliminate_seasonality(self):

        for pond in self.pond_names:
            pond_df = self.ponds[pond]
            #print(pond_df.columns)

            seasonal_cols = ['o2', 'temp', 'o2_sat']
            shifted_cols = [x+'_shft' for x in seasonal_cols]
            twice_a_day_cols = ['meas_sec', 'is_eve', 'meas_lag']
            daily_cols=[ 'pond_no', 'date',   'pressure', 'secchi', 'feed_kg', 'sun_elev',
             'sunrise', 'sunset', 'moon_phase', 'max_temp', 'min_temp', 'avg_temp',
             'sun_hr', 'uv_idx', 'wind_spd', 'wind_dir', 'weath_code', 'precip_mm',
             'humidity', 'cloud_cov', 'dew_pt', 'wind_gust', 'moonshine', 'sun_sec',
             'cloud_desc', 'days_passed']
            pond_df[shifted_cols] = pond_df[seasonal_cols] - pond_df[seasonal_cols].shift(2)
            self.na_shifted_seasonability_rows[pond] = pond_df[pond_df['o2_shft'].isna()]
            pond_df = pond_df.dropna(axis='rows')
            pond_df = pond_df.drop(columns=seasonal_cols)
            self.ponds[pond] = pond_df



    def create_target_var_season(self):
        for pond in self.pond_names:
            pond_df = self.ponds[pond]
            pond_df['y'] = pond_df['o2_shft'].shift(-1)
            pond_df['tmrw_meas_lag'] = pond_df['meas_lag'].shift(-1)
            pond_df['tmrw_moonshine'] = pond_df['moonshine'].shift(-1)
            pond_df = pond_df.dropna()
            #print(f'created target var, cols: {pond_df.columns}')
            self.ponds[pond] = pond_df



    def merge_mor_eve(self):
        for pond in self.pond_names:
            pond_df = self.ponds[pond]
            # Merge morning and evening
            # discard time information, make rows that have evening and morning info for every date
            cols_to_drop = ['meas_sec', 'is_eve']
            split_cols = ['meas_lag', 'o2', 'temp', 'o2_sat', 'date']
            df_morning = pond_df[split_cols][pond_df['is_eve'] == False].copy()
            df_evening = pond_df[split_cols][pond_df['is_eve'] == True].copy()
            df_mor_eve = pd.merge(df_morning, df_evening, on=['date'], suffixes=('_mor', '_eve'))

            # Don't need both morning and evening values, because they are duplicates here. We have only daily data now.
            pond_df = pond_df.loc[pond_df['is_eve'] == True, :]

            pond_df = pond_df.drop(columns=[
                'meas_sec', 'is_eve',
                'meas_lag', 'o2', 'temp', 'o2_sat'
            ])
            pond_df = pd.merge(df_mor_eve, pond_df, on=['date'], how='left')
            self.ponds[pond] = pond_df

    def create_target_var_tmrw_mor(self):
        for pond in self.pond_names:
            pond_df = self.ponds[pond]
            pond_df['y'] = pond_df['o2_mor'].shift(-1)
            pond_df['tmrw_meas_lag_mor'] = pond_df['meas_lag_mor'].shift(-1)
            pond_df = pond_df.dropna()
            #print(f'created target var, cols: {pond_df.columns}')
            self.ponds[pond] = pond_df

    def add_lagged_features(self, exclude=[], lag=1):

        for pond in self.pond_names:
            pond_df = self.ponds[pond]
            original_cols = pond_df.columns

            for lg in range(1, lag + 1):
                for column in original_cols:
                    if column not in exclude:
                        pond_df[f'{column}_lg{lg}'] = pond_df[column].shift(lg)

            pond_df = pond_df.dropna()
            self.ponds[pond] = pond_df



    def drop_cols(self,
                  cols_to_drop: list):
        for pond in self.pond_names:
            self.ponds[pond] = self.ponds[pond].drop(columns=cols_to_drop)
