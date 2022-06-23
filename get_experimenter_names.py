#!/usr/bin/python3
# Get list of experimenters for a particular experiment
# Derek Fujimoto
# June 2022

import numpy as np
import bdata as bd
import os, datetime, re, sys, warnings
from glob import glob
from tqdm import tqdm
warnings.filterwarnings("ignore")

def get_experimenter_names(exp):

    # get path to data
    if 'BNMR_ARCHIVE' in os.environ.keys():
        nmr_path = os.environ['BNMR_ARCHIVE']
    elif os.path.isdir(os.path.join(os.environ['HOME'], '.bdata', 'bnmr')):
        nmr_path = os.path.join(os.environ['HOME'], '.bdata', 'bnmr')
    else:
        raise RuntimeError("I don't know where to find the BNMR files. Set BNMR_ARCHIVE environment variable")
            
    if 'BNQR_ARCHIVE' in os.environ.keys():
        nqr_path = os.environ['BNQR_ARCHIVE']
    elif os.path.isdir(os.path.join(os.environ['HOME'], '.bdata', 'bnqr')):
        nqr_path = os.path.join(os.environ['HOME'], '.bdata', 'bnqr')
    else:
        raise RuntimeError("I don't know where to find the BNQR files. Set BNQR_ARCHIVE environment variable")
        
    # initialize, keyed by year
    experimenters = {}

    # cycle through years
    thisyear = datetime.datetime.now().year
    years = tqdm(range(2010, thisyear+1), total=thisyear-2010+1, leave=False)
    for year in years:
        
        years.set_description(f'Checking {year}')
        ex_thisyear = []
        
        # cycle through nmr and nqr
        for path in (nmr_path, nqr_path):
            
            path = os.path.join(path, str(year), '*.msr')
            
            # get list of all runs
            paths = glob(path)
            runs = (int(os.path.splitext(os.path.basename(p))[0]) for p in paths)
            
            # get all experimenters
            runs_tqdm = tqdm(runs, total=len(paths), leave=False)
            for r in runs_tqdm:          
                
                runs_tqdm.set_description(f'Checking {r}')
                
                try:
                    b = bd.bdata(r, year)
                except Exception:
                    continue
                    
                if b.exp == exp:                
                    ex = b.experimenter
                    ex = re.split(r',| |;|:', ex)
                    ex_thisyear.extend(ex)
                                
        # homoginize
        ex_thisyear = [e.lower().strip() for e in ex_thisyear if e]
        
        # get unique experimenters
        if len(ex_thisyear) > 0:
            experimenters[year] = np.unique(ex_thisyear, return_counts=True)
        
    # print results
    for year in experimenters.keys():
        
        # sort experimenters
        ex = experimenters[year]
        idx = np.argsort(ex[1])
        
        ex = zip(ex[0][idx[::-1]], ex[1][idx[::-1]])
        

        # print table
        print(f'{year}' + '='*20)
        for e, n in ex:
            print(f'{e}\t{n}')
    
    return experimenters
    
    
if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        get_experimenter_names(int(sys.argv[1]))
    else:
        print('usage: get_experimenter_names.py exp')
    
    
    
