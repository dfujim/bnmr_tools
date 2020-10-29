# Read SRIM RANGE.txt and TDATA.txt files into an object
# Derek Fujimoto
# Nov 2018

import pandas as pd
import numpy as np
from scipy.integrate import simps as integrate
import matplotlib.pyplot as plt
import os

# =========================================================================== #
class srim(object):
    """
        Instance variables (RANGE)
            
            avg_range:          range in Ang
            avg_straggle:       stdev in Ang
            avg_range_lat:      lateral range in Ang
            avg_straggle_lat:   lateral stdev in Ang
            avg_range_rad:      radial range in Ang
            avg_straggle_rad:   radial stdev in Ang
            backscatter:        number of backscattered ions
            energy:             implantation energy in keV
            histo:              DataFrame with range histogram
            hist_norm:          Area under histo
            ion_species:        atomic type
            ion_quantity:       number of implanted ions
            target:             target info
            transmitted:        number of transmitted ions
            version:            SRIM version
    """
    
    # ======================================================================= #
    def __init__(self,fetchdir='.',range_txt='RANGE.txt'):
        
        self.get_range(os.path.join(fetchdir,range_txt))
                
    # ======================================================================= #
    def get_range(self,filename):
        
        # read file lines
        with open(filename,'r') as fid:
            lines = fid.readlines()
            
        # set version number
        for i,line in enumerate(lines):
            if 'SRIM-' in line:
                self.version = line.strip()
            elif i>0:
                index = i
                break
                
        index += 11
        # set ion 
        for i,line in enumerate(lines[index:]):
            if i == 0:
                line_spl = line.split()
                self.ion_species = line_spl[2]
                self.energy = float(line_spl[-2])
            elif line[0] == '=': 
                index += i
                break
            
        # set material stats
        s = []
        for i,line in enumerate(lines[index:]):
            
            if line[0] != '=':
                s.append(line.strip())
            elif i>0:
                index += i
                break
        self.target = s
        # set stats of implantation 
        for i,line in enumerate(lines[index:]):
            if i == 1:
                self.ion_quantity = int(float(line.split()[-1].replace('=','')))
            elif i == 2:
                self.avg_range = float(line.split()[4])
                self.avg_straggle = float(line.split()[-2])
            elif i == 3:
                self.avg_range_lat = float(line.split()[4])
                self.avg_straggle_lat = float(line.split()[-2])
            elif i == 4:
                self.avg_range_rad = float(line.split()[4])
                self.avg_straggle_rad = float(line.split()[-2])
            elif line[0] == '=' and i>0:
                index += i
                break
        
        # set transmit,backscatter
        for i,line in enumerate(lines[index:]):
            
            if i==1:
                line_spl = line.split()
                if line_spl[2][1:] != ';':
                    self.transmitted = int(line_spl[2][1:-1])
                else:
                    self.transmitted = 0
                
                if line_spl[-1][1:] != ';':
                    self.backscatter = int(line_spl[-1][1:])
                else:
                    self.backscatter = 0
            elif i>1:
                index += i
                break
        
        index += 12    
        # set depth histo
        depth = []
        ions = []
        for line in lines[index:]:
            l = line.split()
            depth.append(float(l[0]))
            ions.append(float(l[1]))
        
        self.histo = pd.DataFrame({'depth_bin':depth,'ion_count':ions})
        
        # histogram normalization
        self.hist_norm = integrate(self.histo['ion_count'].values,
                                   self.histo['depth_bin'].values)
        
    # ======================================================================= #
    def interp_hist(self,x):
        """Interpolate histogram to get probability values for any depth"""
        
        # get data
        bins = self.histo['depth_bin'].values
        hist = self.histo['ion_count'].values/self.hist_norm

        # interpolate
        return np.interp(x,bins,hist)
    
    # ======================================================================= #
    def draw(self):
        plt.plot(self.histo['depth_bin'],self.histo['ion_count']/self.hist_norm)
