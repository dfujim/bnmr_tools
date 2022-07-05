# Read striptool into pandas dataframe
# Derek Fujimoto
# Jul 2022

import pandas as pd
from io import StringIO

def read_striptool(filename):

    # read file
    with open(filename, 'r') as fid:
        header = fid.readline()
        contents = fid.read()

    # check if csv or dat file
    is_csv = ',' in header
    
    # parse header
    if not is_csv: 
        header = 'Date\t' + header
    header = header.replace(' [', '[')

    # join with contents
    contents = header + contents

    # parse to dataframe
    if is_csv:
        df = pd.read_csv(StringIO(contents), na_values='BadVal')
    else:
        df = pd.read_csv(StringIO(contents), sep='\s+', na_values='BadVal')

    # convert to datetime
    if is_csv:
        df['datetime'] = df.apply(lambda x: f"{x['Time']}", axis='columns')
    else:
        df['datetime'] = df.apply(lambda x: f"{x['Date']} {x['Time']}", axis='columns')
        df.drop(columns='Date', inplace=True)
        
    df['time'] = pd.to_datetime(df['datetime'], format="%m/%d/%Y %H:%M:%S.%f")

    # set as index and drop other columns
    df.set_index('time', inplace=True)
    df.drop(columns=['datetime', 'Time'], inplace=True)

    return df
