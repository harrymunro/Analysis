"""
Created on Wed Dec 14 21:15:48 2016

@author: HarryMunro

NETMIS data is data recorded on the London Underground that mainly consists of train location occupation times.

This code takes in a NETMIS-ish file and resamples by unique location.

This has been created for my MSc project.

"""

# script is currently working to return a dataframe with SUTOR CODES as columns, indexed by a resampled timestamp

# takes in some NETMIS data
# deletes stations in sutor code list
# deletes rows where timestamp data is not available
# calculates dwell times


import pandas as pd
from datetime import datetime

input_csv = 'Populated NETMIS (including days) with RODS.csv' ### MUST BE IN TIME FORMAT yyy/mm/dd hh:mm:ss

#import
df = pd.read_csv(input_csv)

df = df.drop('Unnamed: 0', 1)
df = df.drop('index', 1)

# calculate boarders and alighters
df['BOARDERS AND ALIGHTERS'] = df['BOARDERS'] + df['ALIGHTERS']

# convert timestamp data
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format = '%Y-%m-%d %H:%M:%S')
df = df.sort(columns = 'TIMESTAMP')

# Trimming dwell outliers
#df = df[df['DWELL TIME'] < 45] # optional

# Split into each sutor group
grouped = df.groupby(df['SUTOR DIRECTION LINE'])

# new dataframe
# create columns for dataframe
#columns = ['DWELL TIME', 'LINE ID', 'DIRECTION CODE', 'ALIGHTERS', 'BOARDERS']
df2 = pd.DataFrame()


# Resample for each SUTOR group --- stuck here
for group in df['SUTOR DIRECTION LINE'].unique(): # creates a list of tuples
    sample = grouped.get_group(group)
    if len(sample) > 0.25 / len(df['SUTOR DIRECTION LINE'].unique()) * len(df): # only proceed for "good" data based on size of dataframe and number of unique locations
        sample = sample.set_index('TIMESTAMP') # good
        #sample = sample.sort()
        sample = sample.resample('H', how = 'mean')# .std() # good
        sample['SUTOR DIRECTION LINE'] = group
        #df2 = pd.concat([sample, df2], axis = 1)
        #df2 = df2.rename(columns={'DWELL TIME': group})
        df2 = df2.append(sample)
        df2 = df2.dropna()# remove NAN rows

# clean up dwell times 
df2 = df2[df2['DWELL TIME'] < 45] # less than
# write to csv 
df2.to_csv('resampled_by_unique_location.csv')
