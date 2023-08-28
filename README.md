# OxygenPredictionRF

I've incorporated data on water temperature, Secchi disc visibility, and weather. It's worth noting that the weather data was sourced from a nearby city, which might not perfectly mirror the pond's conditions. I believe that gathering on-site weather data could significantly enhance the model's accuracy. Additionally, integrating weather forecasts could offer even more precision.

The published notebook is here: https://lnkd.in/gVFJH5nt

A surprising revelation was the value of astronomical data. Factors like sun elevation and a specially devised "moonshine" parameter (a combination of moon elevation, moon visibility duration, and the percentage of the moon's illuminated surface) proved beneficial.

The model's outcomes were promising:
Pond np1: RMSE: 1.21, R2: 0.44
Pond np2: RMSE: 1.07, R2: 0.31
Pond vp1: RMSE: 2.15, R2: 0.42
Pond vp2: RMSE: 1.54, R2: 0.36
Pond vp3: RMSE: 1.39, R2: 0.58
Pond vp4: RMSE: 1.24, R2: 0.69

However, the model couldn't predict sudden drops in oxygen. This might be attributed to missing water parameters like pH, nitrogen forms, alkalinity, oxygen demand, and the quality of the weather data.
For perspective, a study titled "Machine Learning for Manually-Measured Water Quality Prediction in Fish Farming" published in PLoS One reported an RMSE of 1.1787 and R2 of 0.63.
