import math
import numpy as np
from pvlib import location
import ephem
import pandas as pd
from datetime import datetime, timedelta


def oxygen_saturation(t, atm_pressure=760):
    """
    Calculate the dissolved oxygen concentration at saturation in water in mg/l.
    Based on formulas in Timmons' Recirculating Aquaculture 5th edition page 363

    Parameters:
    t (float): Temperature in degrees Celsius.
    atm_pressure (float): Atmospheric pressure in mmHg. Default is 760.

    Returns:
    float: Dissolved oxygen concentration at saturation in mg/l.
    """
    # Calculate b using the given formula
    b = math.exp(
        -58.3877
        + 85.8079 * (100 / (273.15 + t))
        + 23.8439 * math.log((273.15 + t) / 100)
    )

    # Calculate water vapor pressure using the given formula
    water_vapor_press = (
            0.61078
            * 7.50062
            * math.exp(17.27 * t / (t + 237.3))
    )

    # Oxygen percentage in atmosphere
    oxygen_percentage = 0.20946

    # Calculate oxygen saturation using the given formula
    oxygen = (
            1000
            * 1.42903
            * b
            * oxygen_percentage
            * ((atm_pressure - water_vapor_press) / atm_pressure)
    )

    return oxygen

# max sun elevation (sin)
def daily_max_elevation(latitude, longitude, start_date, end_date):
    # Define location
    site = location.Location(latitude, longitude)

    # Define date range
    dates = pd.date_range(start_date, end_date, freq='D')

    # Initialize list to store max zenith angles
    max_elevation_angles = []

    # Loop over each date
    for date in dates:
        # Define time range for the day
        times = pd.date_range(date, date + timedelta(days=1), freq='10min')

        # Calculate solar position
        solar_position = site.get_solarposition(times)

        # Calculate max sun elevation angle for the day and its cosine
        elevation = solar_position['apparent_elevation'].apply(lambda x: np.sin(np.deg2rad(x)))
        max_elevation = elevation.max()

        # Append to the list
        max_elevation_angles.append(max_elevation)

    # Convert to pandas DataFrame
    max_elevation_df = pd.DataFrame({'date': dates, 'sun_elev': max_elevation_angles})

    return max_elevation_df

def daily_max_moon_elevation(latitude, longitude, start_date, end_date):
    # Create an observer
    observer = ephem.Observer()

    # Set the observer's latitude and longitude
    observer.lat = str(latitude)
    observer.lon = str(longitude)

    # Define date range
    dates = pd.date_range(start_date, end_date, freq='D',)

    # Initialize list to store max moon elevations
    max_moon_elevations = []

    # Loop over each date
    for date in dates:
        # Define time range for the night
        times = pd.date_range(date, date + timedelta(hours=24), freq='10min')

        # Initialize list to store moon elevations
        moon_elevations = []

        # Loop over each time
        for time in times:
            # Set the observer's date and time
            observer.date = time.to_pydatetime()

            # Create a moon object
            moon = ephem.Moon()

            # Compute the moon's position
            moon.compute(observer)

            # Calculate the moon's elevation
            moon_elevation = np.sin(moon.alt)

            # Append to the list
            moon_elevations.append(moon_elevation)

        # Calculate max moon elevation for the night
        max_moon_elevation = max(moon_elevations)

        # Append to the list
        max_moon_elevations.append(max_moon_elevation)

    # Convert to pandas DataFrame
    max_moon_elevation_df = pd.DataFrame({
        'date': dates,
        'moon_elev': max_moon_elevations
    })

    return max_moon_elevation_df





if __name__ == '__main__':
    # List of temperatures from 0 to 30 degrees Celsius
    temps = list(range(31))

    # Calculate and print oxygen saturation for each temperature
    print([oxygen_saturation(t) for t in temps])

    # Test the function
    latitude, longitude = 55.876856, 48.063189
    start_date = '2023-05-10'
    end_date = '2023-07-23'

    max_moon_elevation_df = daily_max_moon_elevation(latitude, longitude, start_date, end_date)
    print(max_moon_elevation_df)
