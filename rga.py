# Draw RGA output files
# Derek Fujimoto
# Nov 2020

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime, io

class rga(object):
    
    """
        Data fields: 
            
            filetype:       string, one of "histogram", "PvT"
            data:           DataFrame or Series
            table:          PvT only, channel characteristics
    """
    
    def __init__(self, filename):
        
        # read the file 
        with open(filename, 'r') as fid: 
            lines = fid.readlines()
        
        # figure out what kind of file it is
        self.filetype = None
        for l in lines: 
            
            if 'Histogram' in l:   
                self.filetype = 'histogram'
                self._read_histogram(lines)
                break
            elif 'Pressure vs Time' in l: 
                self.filetype = 'PvT'
                self._read_pvt(lines)
                break
        
        if self.filetype is None:
            raise RuntimeError("Unknown file type")
            
    def _draw_hist(self, ax):
        
        # draw histogram as bar graph
        x = self.data.index
        y = self.data.values
        
        ax.bar(x,y)
        ax.set_xlabel("Mass (%s)" % self.header['Scan Start Mass'][1])
        ax.set_ylabel('Pressure (%s)' % self.header['Units'][0])
        ax.set_yscale('log')
        
    def _draw_pvt(self, ax):
        
        # sort the columns
        avg = self.data.mean(axis='index')
        idx = np.argsort(avg)[::-1]
        cols = self.data.columns[idx]
        
        for c in cols:    
            ax.plot(self.data.index/3600, self.data[c], 
                    label=c+' (%g amu)' % self.table.loc[c, 'Mass(amu)'])
            
        ax.legend(fontsize='xx-small')
        ax.set_xlabel("Time (h)")
        ax.set_ylabel('Pressure (%s)' % self.header['Units'][0])
        ax.set_yscale('log')
            
    def _read_header(self, lines):
        """
            Read the header lines to a dictionary
        """
        self.header = {}
        for l in lines:
            
            # clean
            l = l.strip()
            if not l: continue
            
            # get header info
            info = l.split(',')
            
            if len(info) < 3:       unit = None
            else:                   unit = info[2].strip()
                
            val = info[1].strip()
            key = info[0].strip()
            
            self.header[key] = (val, unit)
            
    def _read_histogram(self, lines):
        """
            Read histogram file
        """
        
        # read the header
        self._read_header(lines[6:20])
        
        # read the data
        dlines = ''.join(lines[21:])
        names = (self.header['Scan Start Mass'][1], 
                 self.header['Units'][0])
        self.data = pd.read_csv(io.StringIO(dlines), names=names)
        self.data.set_index(names[0], inplace=True)
        
        self.data = pd.Series(self.data[names[1]])
        
    def _read_pvt(self, lines):
        """
            Read P v T file
        """
        
        # read the header
        self._read_header(lines[6:17])
        
        # read the channel table 
        nchannels = int(self.header['Active channels in Scan'][0])
            
        table_header = tuple(map(str.strip, lines[17].split(',')))
        table_lines = lines[19:19+nchannels]
        self.table = pd.read_table(io.StringIO(''.join(table_lines)), 
                                    names=table_header, sep='\s+')
        self.table.set_index('Channel', inplace=True)
        
        # read the data
        del lines[27]
        lines[26] = ','.join(lines[26].split())+',\n'
        
        data_lines = ''.join(lines[26:])
        self.data = pd.read_csv(io.StringIO(data_lines))
        self.data.set_index('Time(s)', inplace=True)

        # drop empty columns
        drop_c = [c for c in self.data.columns if 'Unnamed' in c]
        for c in drop_c:        
            self.data.drop(c, axis='columns', inplace=True)
        
        # replace the channel names
        new_names = {'Channel#%d'%i: self.table.loc[i,'Name'] for i in self.table.index}
        self.data.rename(new_names, inplace=True, axis='columns')

        # reset table
        self.table.set_index('Name', inplace=True)

    def draw(self, ax = None):
        """Draw the file"""
        
        new_ax = ax is None
        
        if new_ax: 
            plt.figure()
            ax = plt.gca()
        
        if self.filetype == 'histogram':
            self._draw_hist(ax)
        elif self.filetype == 'PvT':
            self._draw_pvt(ax)

        if new_ax: 
            plt.tight_layout()
