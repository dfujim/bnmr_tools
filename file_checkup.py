#!/usr/bin/python3
# Derek Fujimoto
# Aug 2018

import os,datetime,sys
from bdata import bdata
from tabulate import tabulate

lst_nmr = {
    'camp':[
        '/Magnet/mag_field'         ,
        '/Cryo_level/He_level'      ,
        '/Cryo_level/N2_level'      ,
        '/CryoEx_MassFlow/read_flow',
        '/Dewar/He_level'           ,
        '/Magnet/mag_read'          ,
        '/mass_flow/read_flow'      ,
        '/Needle/set_position'      ,
        '/rf_level_cont/dac_set'    ,
        '/rfamp/RF_gain'            ,
        '/Sample/current_read_1'    ,
        '/Sample/read_A'            ,
        '/Sample/read_B'            ,
        '/Sample/read_C'            ,
        '/Sample/setpoint_1'        , 
        '/PVac/adc_read'            ,
        ],
    'epics':[
        'ILE2:BIAS15:RDVOL'         ,
        'ILE2:BIASTUBE:VOL'         ,
        'ILE2:LAS:RDPOWER'          ,
        'BNMR:HVBIAS:POS:RDVOL'     ,
        'BNMR:HVBIAS:NEG:RDVOL'     ,
        'ITE:BIAS:RDVOL'            ,
        'ITW:BIAS:RDVOL'            ,
        ],
    'ppg':[
        'PPG/PPG1f/constant time between cycles',
        'PPG/PPG1f/use defaults for midbnmr',
        'PPG/PPG1f/Bin width (ms)',
        'PPG/PPG1f/psm fREF enabled',
        'PPG/PPG1f/psm fREF scale factor',
        'PPG/PPG1f/frequency increment (Hz)',
        'PPG/PPG1f/frequency start (Hz)',
        'PPG/PPG1f/frequency stop (Hz)',
        'PPG/PPG1f/Enable helicity flipping',
        'PPG/PPG1f/Helicity flip sleep(ms)',
        'PPG/PPG1f/init mode file',
        'PPG/PPG1f/PPG mode',
        'PPG/PPG1f/num bins',
        'PPG/PPG1f/num cycles per supercycle',
        'PPG/PPG1f/number of midbnmr regions',
        'PPG/PPG1f/psm onef enabled',
        'PPG/PPG1f/psm onef scale factor',
        'PPG/PPG1f/DAQ service time (ms)',
        ]
}

lst_nqr = {
    'camp':[
        '/cryo_lift/read_position',
        '/cryo_lift/set_position',
        '/He_flow/read_flow',
        '/He_flow/set_flow',
        '/needle-valve/read_position',
        '/needle-valve/set_position',
        '/biasV/output1',
        '/Sample/current_read',
        '/Sample/read_A',
        '/Sample/read_B',
        '/Sample/setpoint',
        '/Sample/set_current',
        ],
    'epics':[
        'ILE2:BIAS15:RDVOL',
        'ILE2A1:HH:RDCUR',
        'ILE2:LAS:RDPOWER',
        'BNQR:HVBIAS:RDVOL',
        'ITW:BIAS:RDVOL',
        'ITE:BIAS:RDVOL',
        ],
    'ppg':[
        'PPG/PPG20/e20 beam off dwelltimes',
        'PPG/PPG20/e20 beam on dwelltimes',
        'PPG/PPG20/Dwell time (ms)',
        'PPG/PPG20/e20 rf frequency (Hz)',
        'PPG/PPG20/Enable helicity flipping',
        'PPG/PPG20/helicity flip sleep (ms)',
        'PPG/PPG20/init mode',
        'PPG/PPG20/PPG mode',
        'PPG/PPG20/e20 prebeam dwelltimes',
        'PPG/PPG20/RFon duration (dwelltimes)',
        'PPG/PPG20/RFon delay (dwelltimes)',
        'PPG/PPG20/DAQ drives sampleref',
        'PPG/PPG20/enable sampleref mode',
        ]
}


def file_checkup(area='bnmr',run=-1,year=-1):
    """
        Check that all variables are being logged in the latest file. 
    """
    
    # set inputs -------------------------------------------------------------
    
    # get list of needed inputs
    area_lst = lst_nmr if area == 'bnmr' else lst_nqr
    
    # get data directory
    dirt = os.environ[area.upper()+'_ARCHIVE']
    if dirt[-1] != '/': dirt+='/'
    
    # get latest year with data
    if year < 0:
        year = datetime.datetime.now().year
        while str(year) not in os.listdir(dirt):
            year -= 1
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
        l = area_lst[tk]
        
        camp = []
        for k in keys:
            v = lst[k]
            camp.append([v.title,v.mean,v.units])
            
            try:
                l.pop(l.index(v.title))
            except ValueError as errmsg:
                print(errmsg)
        
        print(tk.upper()+' VARIABLES ')#+('='*60))
        print(tabulate(camp,headers=['Quantity','Value','Units']),end='\n\n')
        print('Missing: ')
        for l_missing in l:
            print('\033[91m\t',l_missing,'\033[0m')
        if len(l) == 0: print('\tNone')
        print('\n')
        
    # histogram info ---------------------------------------------------------
    hist = []
    keys = list(bd.hist.keys())
    keys.sort()
    for k in keys:
        if not ('-' in k or '+' in k): continue
            
        h = bd.hist[k]
        hist.append([h.title,np.sum(h.data)/bd.duration])
            
    print('HISTOGRAMS ')#+('='*60))
    print(tabulate(hist,headers=['Histogram','Rate (1/s)']),end='\n\n')

# run if main
if __name__ == '__main__':
    
    args = sys.argv[1:]
    
    
    try:
        area = args[0]
    except IndexError:
        area = 'both'
    
    try:
        run = int(args[1])
    except IndexError:
        run = -1
        
    try:
        year = int(args[2])
    except IndexError:
        year = -1
    
    if area == 'both':
        file_checkup('bnmr')
        file_checkup('bnqr')
    else:
        file_checkup(area,run,year)
