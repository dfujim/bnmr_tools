# Draw campvarlog files
# Derek Fujimoto
# Mar 2020

import pandas as pd
import numpy as np
import bdata as bd
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl

class campvarlog(object):
    """
        df: dataframe with data
    """
    
    def __init__(self,filename):
        
        # read the file 
        with open(filename,'r') as fid:
            file_contents = fid.read()
            
        # split by header line
        files = file_contents.split('!')
        
        # process each file
        df_list = []
        for file_contents in files:
  
            # blank line
            if not file_contents:
                continue
            
            lines = file_contents.split('\n')
        
            # get the header
            header = lines[0]
        
            # remove blank linkes
            lines = [l for l in lines if l]            
            
            # split and strip
            header = header.strip().split()
            lines = [l.strip().split() for l in lines[1:]]
        
            # make data frame    
            df = pd.DataFrame(lines,columns=header).astype(float)
            
            # set time as index
            df.loc[:,'time'] = df['time'].apply(datetime.datetime.fromtimestamp)
            df['duration'] = df.loc[:,'time']-df.loc[0,'time']
            df.set_index('time',inplace=True)
            
            # save the data frame
            df_list.append(df)
        
        # combine the dataframes
        df = pd.concat(df_list)
        
        # convert dataframe columns to bdata standards
        conv = {key:name for key,name in bd.bdata.dkeys.items() if key in df.columns}
        df.rename(columns=conv,inplace=True)
        
        # save the data
        self.df = df
        self.columns = df.columns.tolist()
    
    def draw(self,ycolumn,xcolumn='time'):
        
        # get x data
        if xcolumn == 'time':       x = self.df.index
        else:                       x = self.df[xcolumn]
        
        # get ydata
        if ycolumn == 'time':       y = self.df.index
        else:                       y = self.df[ycolumn]
        
        # draw
        plt.plot(x,y,label=ycolumn)
        
        # format for timedelta (default is in ns)
        if xcolumn == 'duration':
            formatter = mpl.ticker.FuncFormatter(lambda x,pos: str(datetime.timedelta(seconds=x*1e-9)))
            plt.gca().xaxis.set_major_formatter(formatter)
            
        # plot details
        plt.xlabel(xcolumn)
        plt.legend()
        plt.tight_layout()
        
