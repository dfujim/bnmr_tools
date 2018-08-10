#!/usr/bin/python3
# Derek Fujimoto
# Aug 2018

import os,datetime,sys
from bdata import bdata
from tabulate import tabulate

def file_checkup(area='bnmr',run=-1,year=-1):
    """
        Check that all variables are being logged in the latest file. 
    """
    
    # set inputs -------------------------------------------------------------
    
    # get data directory
    dirt = os.environ[area.upper()+'_ARCHIVE']
    
    # get latest year
    if year < 0:
        year = max(map(int,os.listdir(dirt)))
        dirt += '%d/' % year 
    
    # get latest run 
    if(run < 0):
        files = os.listdir(dirt)
        files.sort()
        run = int(os.path.splitext(files[-1])[0])
    
    # open data file
    bd = bdata(run,year)

    # print header -----------------------------------------------------------
    print('\nDIAGNOSTICS FOR RUN %d, %d ' % (run,year) + "="*43,end='\n\n')

    # check set and read variables -------------------------------------------
    top_keys = ['camp','epics','ppg']
    
    for lst,tk in zip([bd.camp,bd.epics,bd.ppg],top_keys):
    
        keys = list(lst.keys())
        keys.sort()
        
        camp = []
        for k in keys:
            v = lst[k]
            camp.append([v.title,v.mean,v.units])
        
        print(tk.upper()+' VARIABLES '+('='*60))
        print(tabulate(camp,headers=['Quantity','Value','Units']),end='\n\n')
        
    # histogram info ---------------------------------------------------------
    hist = []
    keys = list(bd.hist.keys())
    keys.sort()
    for k in keys:
        if not ('-' in k or '+' in k): continue
            
        h = bd.hist[k]
        hist.append([h.title,np.sum(h.data)/bd.duration])
            
    print('HISTOGRAMS '+('='*60))
    print(tabulate(hist,headers=['Histogram','Rate (1/s)']),end='\n\n')


# run if main
if __name__ == '__main__':
    
    args = sys.argv[1:]
    
    
    try:
        area = args[0]
    except IndexError:
        area = 'bnmr'
    
    try:
        run = int(args[1])
    except IndexError:
        run = -1
        
    try:
        year = int(args[2])
    except IndexError:
        year = -1
    
    file_checkup(area,run,year)
