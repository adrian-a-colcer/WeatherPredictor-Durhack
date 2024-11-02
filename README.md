# About The Project
This is a project developed at durhack, a hackathon hosted by Durham University. The challenge for this project is 'chain of events' by Marshall Wace. My idea for this hackathon is a weather predictor using time series forecasting, a technique that uses historical data (or a chain of events) to predict future events. Time series forecasting is also used in meteorology, econometrics, quantitative finance, and statistics. Predictions will be made using the Long Short-Term Memory model (LSTM) because it has been shown that it excels at predicting sequential data, weather data is sequential in nature. Weather data is sourced from Durham University's observatory under the Daily Climate Obervations [Dataset](https://durhamweather.webspace.durham.ac.uk/open-access-climate-datasets/) [[1]](#1).

## The weather dataset
This dataset contains 10 usable features: Tmax °C, Tmin °C, Tmean °C, Tgmin °C, Air frost, Grd frost, Days 25 °C, Pptn mm, Rain day and Wet day. There are additional features in the excel file, however this data was not recorded consistently. This data was collected daily between 23/7/1843 to 30/4/2022, we will only use the last 20 years to reduce training time. 

## Feature Engineering
Before the model accepts any features we need to format this data. When we access read the excel sheet we need to make sure the correct one is selected, hence the sheet_name value. First we remove the unnecessary data from the dataframe. Data recorded before 2002 and empty columns are removed:

```python
import pandas as pd

filename = "Durham-Observatory-daily-climatological-dataset.xlsx"
df = pd.read_excel(filename,sheet_name = "Durham Obsy - daily data 1843-")
df.drop(df[df["YYYY"] < 2002].index, inplace=True)
df.drop(df.columns[14:], axis = 1, inplace=True)
```
Now if we check head of the dataset we will see entries from 2002 onwards, and the fields are the features that we need:

```python
df.head()
```

![Alt text](images/DatasetHead.PNG "First five rows of data set")


The date is a feature we need to format because it has daily and yearly periodicity.



## References
<a id="1">[1]</a>
Burt, Stephen and Burt, Tim, 2022. Durham Weather and Climate since 1841. Oxford University Press, 580 pp.

