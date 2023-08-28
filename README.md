# OxygenPrediction

In my recent exploration into the world of aquaculture data science, I delved deep into the data from my fish ponds. The challenge? Most up-to-date research utilizes data that is constantly gathered with submerged DO probes. In contrast, our measurements were taken manually, twice a day, during the morning and evening, closely aligning with sunrise and sunset timings (with a variance of +/- one hour).

## SARIMA model

In this research (Arima.ipynb) , I focused on using past DO measurements to predict future DO levels. The goal was clear: predict DO utilizing solely DO data.

TLDR: The model I developed accurately captures the overarching trends, presenting an RMSE ranging from 0.8 to 2. It excels when the data remains stable. However, during heightened DO fluctuations, especially those induced by liming or the introduction of manure, the model faces challenges.

While the current model offers promising results, there's room for improvement. Incorporating additional data such as water temperature, the amount of feed given, weather data, and secchi disc visibility can potentially enhance the model's accuracy and adaptability.

For context, a previous research titled "Machine learning for manually-measured water quality prediction in fish farming" published in PLoS One achieved an RMSE of 1.1787. The study, led by Andres Felipe Zambrano and team, highlighted the potential of machine learning in scenarios with limited data availability in aquaculture. You can read the full research here: https://lnkd.in/gTNTtaTe

For those interested in my methodology and results:
Published Version: https://lnkd.in/gYzaMhfQ

Full Notebook with Code: https://lnkd.in/gk5FxAjF

Your insights, feedback, and collaboration are always welcome. Let's continue to harness the power of data for sustainable aquaculture!

## Random Forest

This time around, I've incorporated not just past DO readings but also data on water temperature, Secchi disc visibility, and weather. It's worth noting that the weather data was sourced from a nearby city, which might not perfectly mirror the pond's conditions. I believe that gathering on-site weather data could significantly enhance the model's accuracy. Additionally, integrating weather forecasts could offer even more precision.

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
