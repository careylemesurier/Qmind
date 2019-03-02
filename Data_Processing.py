import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime

'''
A program that creates scaled testing data out of a given csv file
'''

# Load training data, sorted by company
data_df = pd.read_csv("prices.csv")
# Take only data of desired company
company = "A"
indices = np.where(data_df.values[:, 1] == company)
company_data_df = data_df.iloc[indices[0], :]
# Plot data
plt.figure(figsize=(18, 9))
plt.plot(range(company_data_df.shape[0]), (company_data_df['high'] + company_data_df['close']) / 2.0)
plt.xticks(range(0, company_data_df.shape[0], 1762))
plt.xlabel('Date', fontsize=18)
plt.ylabel('Mid Price', fontsize=18)
plt.show()

# Take average of high and lows
mid_prices = (company_data_df.loc[:, 'high'].values + company_data_df.loc[:, 'low'].values) / 2.0
# Take training data as first half of rows, test data as other half
training_data = mid_prices[:int(mid_prices.shape[0] / 2)]
test_data = mid_prices[int(mid_prices.shape[0] / 2):int(mid_prices.shape[0])]
# Create scaler for data
scaler = MinMaxScaler(feature_range=(0, 1))
# Reshape data for use in scaler
training_data = training_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)

# Smooth data and feed into scaler
smoothing_window_size = int(training_data.shape[0] / 4)
for interval_start in range(0, smoothing_window_size * 4, smoothing_window_size):
    interval_end = interval_start + smoothing_window_size
    scaler.fit(training_data[interval_start:interval_end, :])
    training_data[interval_start:interval_end, :] = scaler.transform(training_data[interval_start:interval_end, :])
# Normalize data not captured in windows
scaler.fit(training_data[interval_start:, :])
training_data[interval_start:, :] = scaler.transform(training_data[interval_start:, :])

# Reshape data
training_data = training_data.reshape(-1)
test_data = scaler.transform(test_data).reshape(-1)

np.savetxt('scaled_training_data.csv', training_data, delimiter=',')
np.savetxt('scaled_test_data.csv', test_data, delimiter=',')