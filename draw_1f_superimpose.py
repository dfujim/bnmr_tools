# Draw Matched 1F Runs
# Derek Fujimoto
# April 2019

import bdata as bd
import matplotlib.pyplot as plt

def draw(run,year,rebin=1,freq_shift=True,base_shift=True,label='run'):
    """
        runlist:    list of runs to draw or int
        yearlist:   list of years corresponding to run, or int
        rebin:      rebin factor, list or int
        freq_shift: if true, shift frequencies to ppm, fixed to window center
        base_shift: if true, baseline shift to zero
        label:      bdata attribute to set to label
    """
    
    # make run list
    if type(run) is int:
        run = [run]
    
    # make year list 
    if type(year) is int:
        year = [year]*len(run)
    
    # make rebin list 
    if type(rebin) is int:
        rebin = [rebin]*len(run)
        
    # get data 
    data = [bd.bdata(r,y) for r,y in zip(run,year)]
    
    # draw
    plt.figure()
    
    for d,rb in zip(data,rebin):
        
        # get asymmetry
        f,a,da = d.asym('c',rebin=rb)
        
        # get baseline shift
        if base_shift:
            bshift = a[0]
        else:
            bshift = 0
            
        # get x values
        if freq_shift:
            fshift = np.mean(f)
            x = (f-fshift)/fshift*1e6
        else:
            x = f*1e-6
        
        # draw
        # ~ plt.errorbar(x,a-bshift,da,fmt='.',label=getattr(d,label))
        plt.plot(x,a-bshift,label=getattr(d,label))
    
    # plot elements
    if base_shift:  plt.ylabel(r'Asym-Asym($\nu_\mathrm{min}$)')
    else:           plt.ylabel('Asymmetry')
    
    if freq_shift:  plt.xlabel('Frequency Shifted to Window Center (PPM)')
    else:           plt.xlabel('Frequency (MHz)')

    plt.legend(fontsize='x-small')
