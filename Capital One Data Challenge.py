# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#########################################################################################################

'''
Import necessary libraries
'''

global fig    #To keep track of figure counts
fig = 0

import collections
import pandas as pd
#Import module for plotting
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import normalize

#########################################################################################################

'''
1. Importing the given data
'''

path = r'E:\Users\Dell\Desktop\green_tripdata_2015-09.csv'    #Change this to local path where data is stored
df = pd.read_csv(path)          #df is a DataFrame
#The DataFrame 'df' has the given data

#########################################################################################################

'''
2. Plotting the histogram of the trip distance ("Trip Distance")
'''

plt.figure(fig)
fig = fig + 1
#Number of bins are taken as 300. The minimum Trip_distance is 0 and maximum is 603.10000000000002
counts, bins, bars = plt.hist(df['Trip_distance'], bins=300, color='black', label='Trip_distance')
plt.legend(loc='upper right')
plt.xlabel('Trip Distance')
plt.ylabel('Counts in the Corresponding Bins')
plt.title('Histogram of Trip_distance (bins = 300)')
plt.savefig('1.Histogram of the trip distance.png', dpi=900, bbox_inches='tight')

#########################################################################################################

'''
3. Report mean and median trip distance grouped by hour of day.
'''

#I have considered the pick up datetime as the time to work upon here, analysing on hourly basis
df['pickup_times'] = pd.to_datetime(df['lpep_pickup_datetime'])
#extract the hour counterpart
df['hours'] = pd.DatetimeIndex(df['pickup_times']).hour

#Temporary dataframe used, not to tamper the original dataframe
temp1 = df

#Sorting by the 'hours' column, to facilitate the calculation of average
temp1 = temp1.sort_values(['hours'], ascending=True)
temp1 = temp1.reset_index()
hours_dict = dict(collections.Counter(temp1['hours']))

dist_avg_by_hour = []
dist_median_by_hour = []
j = 0

#Due to division by 0, some have 'inf' value. Replacing that with '0'
#Averaging Calculation
for i in hours_dict.values():
    temp = temp1['Trip_distance'][j:(j+i)].replace(np.inf, 0)
    dist_avg_by_hour.append(np.mean(temp))
    dist_median_by_hour.append(np.median(temp))
    j = j + i
#The list 'dist_avg_by_hour' represents average trip distance as a function of each hour of day

#Plotting average trip distance each hour for better visualization
plt.figure(fig)
fig = fig + 1
plt.stem(dist_avg_by_hour)
plt.xlabel('Time of the day')
plt.ylabel('Average trip distance in the hour')
plt.margins(0.1, 0.1)
plt.title('Average trip distance by hour')
plt.savefig('2.Average trip distance by hour.png', dpi=900, bbox_inches='tight')

#Plotting median trip distance each hour for better visualization
plt.figure(fig)
fig = fig + 1
plt.stem(dist_median_by_hour)
plt.xlabel('Time of the day')
plt.ylabel('Median trip distance in the hour')
plt.margins(0.1, 0.1)
plt.title('Median trip distance by hour')
plt.savefig('3.Median trip distance by hour.png', dpi=900, bbox_inches='tight')

#########################################################################################################

'''
4. Build a derived variable for tip_out_of_total_fare, 
   which is tip as a percentage of the total fare.
'''

#Add a new column named 'tip_out_of_total_fare' to the DataFrame 'df'
df['tip_out_of_total_fare'] = (df['Tip_amount'] / df['Total_amount']) * 100
#The column 'tip_out_of_total_fare' of 'df' DataFrame has the tip as a percentage of the total fare.

'''
4. Build a predictive model for tip as a percentage 
   of the total fare. Use as much of the data as you 
   like (or all of it). We will validate a sample.
'''

#makiing a temporary duplicate
temp = df

#Extract Predictions
train_targets = np.nan_to_num(temp['tip_out_of_total_fare'])

#Delete unnecesary columns that lack the capability to predict
#Tip amount is also excluded, and if its kept, prediction remains a mere division problem!
columns_to_keep = ['RateCodeID', 'Pickup_longitude', 'Pickup_latitude', 
                   'Dropoff_longitude', 'Dropoff_latitude', 'Passenger_count',
                   'Trip_distance', 'Fare_amount', 'Extra', 'improvement_surcharge',
                   'Total_amount', 'Payment_type', 'Trip_type ', 'MTA_tax']
temp = temp[columns_to_keep]

#To make every sample, a unit vector. This is due to different ranges of feature values
temp = normalize(np.nan_to_num(temp), norm='l2', axis=1)

#Replace all the nan values with '0' 
train = np.nan_to_num(np.array(temp))

#fit a linear regression model
model_LR = LinearRegression()
model_LR.fit(train, train_targets)
'''
'model_LR' is the required model
Now, for validating, put the data in exact order as 'columns_to_keep',
and then assign that to to_be_tested. Then un-comment the following line
model_LR.predict(normalize(to_be_tested), norm='l2', axis=1)
'''

#########################################################################################################

'''
5. Option A1: Build a derived variable representing the average speed over the course of a trip.
'''

#Calculated the total time (in seconds)
df['time_taken'] = (pd.to_datetime(df['Lpep_dropoff_datetime']) - pd.to_datetime(df['lpep_pickup_datetime'])).astype('timedelta64[s]')
#convert seconds to hours
df['time_taken'] =df['time_taken'] / 3600     
#The column 'time_taken' of 'df' DataFrame has the average speed over the course of a trip.

#Average speed = Total distance / Total Time Taken
#I have calculated speed in miles per hours
df['Avg_speed'] = df['Trip_distance'] / df['time_taken']
#The column 'Avg_speed' of 'df' DataFrame has the average speed over the course of a trip.


'''
5. Option A2: Perform a test to determine if the average trip speeds are 
materially the same in all weeks of September. If you decide they 
are not the same, can you form a hypothesis regarding why they differ?
'''

#I have considered the pick up datetime as the time to work upon here
df['pickup_times'] = pd.to_datetime(df['lpep_pickup_datetime'])
#extract the day counterpart
df['day'] = pd.DatetimeIndex(df['pickup_times']).day
day_dict = dict(collections.Counter(df['day']))
#day_dict represents number of rides per day
days_in_weeks = [5, 7, 7, 7, 4]   #number of days in each week in september 2015
j = 0
weektrips = []

#For calculating trips in each week
for i in days_in_weeks:
    weektrips.append(np.sum(day_dict.values()[j:(j+i)]))
    j = j + i

j = 0
avg_by_week = []
#Due to division by 0, some have 'inf' value. Replacing that with '0'
#Averaging Calculation
for i in weektrips:
    temp = df['Avg_speed'][j:(j+i)].replace(np.inf, 0)
    avg_by_week.append(np.mean(temp))
    j = j + i
#The list 'avg_by_week' represents average trip speed as a function of week

labels = ['First', 'Second', 'Third', 'Fourth', 'Fifth']

plt.figure(fig)
fig = fig + 1
plt.stem(avg_by_week)
plt.xlabel('Weeks')
plt.ylabel('Average trip speeds in the week')
plt.xticks([0, 1, 2, 3, 4], labels)
plt.margins(0.1, 0.1)
plt.ylim([13, 17])
plt.title('Average trip speeds by week')
plt.savefig('4.Average trip speeds by week.png', dpi=900, bbox_inches='tight')


'''
5. Option A3: Build a hypothesis of average trip speed as a function of time of day
'''


#I have considered the pick up datetime as the time to work upon here, analysing on hourly basis
df['pickup_times'] = pd.to_datetime(df['lpep_pickup_datetime'])
#extract the hour counterpart
df['hours'] = pd.DatetimeIndex(df['pickup_times']).hour

#Temporary dataframe used, not to tamper the original dataframe
temp1 = df

#Sorting by the 'hours' column, to facilitate the calculation of average
temp1 = temp1.sort_values(['hours'], ascending=True)
temp1 = temp1.reset_index()
hours_dict = dict(collections.Counter(temp1['hours']))

avg_by_hour = []
j = 0

#Due to division by 0, some have 'inf' value. Replacing that with '0'
#Averaging Calculation
for i in hours_dict.values():
    temp = temp1['Avg_speed'][j:(j+i)].replace(np.inf, 0)
    avg_by_hour.append(np.mean(temp))
    j = j + i
#The list 'avg_by_hour' represents average trip speed as a function of each hour of day

#Plotting for better visualization
plt.figure(fig)
fig = fig + 1
plt.stem(avg_by_hour)
plt.xlabel('Time of the day')
plt.ylabel('Average trip speeds in the hour')
plt.margins(0.1, 0.1)
plt.title('Average trip speeds by hour')
plt.savefig('5.Average trip speeds by hour.png', dpi=900, bbox_inches='tight')

#########################################################################################################