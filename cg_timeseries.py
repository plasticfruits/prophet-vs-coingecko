#########################
# %% -- Libraries
#########################

from pycoingecko import CoinGeckoAPI
from prophet.plot import plot_plotly, plot_components_plotly, add_changepoints_to_plot
from prophet import Prophet

# from utils import print_errors
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#########################
# %% -- Import
#########################
# get todays date
today_date = pd.Timestamp.today()
today_unix = (today_date - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

# Define CoinGeckoAPI
cg = CoinGeckoAPI()

# get historical data
get_data = cg.get_coin_market_chart_range_by_id(
    id="bitcoin",
    vs_currency="usd",
    from_timestamp=1328054400,  # 01/01/2012
    to_timestamp=today_unix,
)

# create a for loop to clean data
pull_cols = ["prices", "market_caps", "total_volumes"]
frames = {}

for i in pull_cols:
    x = pd.DataFrame(get_data[i])
    x.rename(columns={0: "date", 1: i}, inplace=True)
    frames[i] = x

# merge DF's
data = pd.merge(frames["prices"], frames["market_caps"], on="date")
data = pd.merge(data, frames["total_volumes"], on="date")

# parse date from UNIX
data["date"] = pd.to_datetime(data["date"], origin="unix", unit="ms")


data.info()
data.head()


#########################
# %% -- Prophet
#########################

# rename cols for Prophet convention
data = data.rename(columns={"date": "ds", "prices": "y"})

# create test df
dates = pd.date_range(start="2022-02-02", end="2022-03-31", freq="D")
dates_df = pd.DataFrame({"ds": dates})

# split train / test
# X_train = data[data.ds < "2022-01-01"][["ds", "y"]]
X_train = data.query("ds > '2019-01-01' & ds < '2022-05-01'")
X_test = dates_df

# fit
model = Prophet(daily_seasonality=True, yearly_seasonality=T)
model.fit(X_train)

# check the predictions for the training data
pred_train = model.predict(X_train)

# use the trained model to make a forecast
pred_test = model.predict(X_test)

# plot forecast
model.plot(pred_test)

# Plot predictions
plot_plotly(model, pred_train)


# %% Add custom events


price_record = pd.DataFrame(
    {
        "holiday": [
            "price_record",
            "price_record",
            "price_record",
            "price_record",
            "price_record",
        ],
        "ds": pd.to_datetime(
            ["18-12-2017", "28-11-2017", "13-10-2017", "04-01-2021", "08-03-2021"]
        ),
        "lower_window": 0,
        "upper_window": 0,
    }
)

liquidation = pd.DataFrame(
    {  # 15% - 25%
        "holiday": "liquidation",
        "ds": pd.to_datetime(
            [
                "27-06-2019",
                "16-07-2019",
                "24-09-2019",
                "26-11-2020",
                "04-01-2021",
                "21-01-2021",
                "23-02-2021",
                "12-05-2021",
                "17-05-2019",
                "05-02-2018",
                "20-12-2017",
                "08-12-2017",
                "14-09-2017",
                "08-01-2015",
                "07-09-2021",
            ]
        ),
        "lower_window": 0,
        "upper_window": 0,
    }
)

super_liquidation = pd.DataFrame(
    {  # > 25%
        "holiday": "super_liquidation",
        "ds": pd.to_datetime(
            ["19-05-2021", "11-01-2021", "12-03-2020", "16-01-2018", "14-01-2015"]
        ),
        "lower_window": 0,
        "upper_window": 0,
    }
)

# concat DF's with custom tags
custom_events = pd.concat([price_record, liquidation, super_liquidation])

m3 = Prophet(holidays=custom_events)
m3.fit(X_train)
pred_train = m3.predict(X_train)
pred_test = m3.predict(X_test)
fig = m3.plot(pred_test)


#########################
# %% -- Temporal Fusion Transformer
#########################
