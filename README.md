# Misc BNMR Python Tools

* [`beta_averaging`](#beta_averaging)
* [`file_checkup`](#file_checkup)
* [`parse_bnmroffice_fit`](#parse_bnmroffice_fit)
* [`draw_1f_superimpose`](#draw_1f_superimpose)
* [`get_experimenter_names.py`](#get_experimenter_names)
* [`campvarlog`](#campvarlog)
* [`read_striptool`](#read_striptool)
* [`rga`](#rga)
* [`srim`](#srim)

### beta_averaging

Code originally written by rmlm for T1 average with uncorrelated errors.

Functions:

```python
t1avg(T1, beta)                       # t1 avg
dt1avg(T1, dT1, beta, dbeta)          # error
```

### file_checkup

File inspector for start of beam. Make sure variables are logged properly.

Functions:

```python

file_checkup(area='bnmr',run=-1,year=-1)
```

As a command line tool:

```
python3 file_checkup [area] [run] [year]
```

### parse_bnmroffice_fit

Turn bnmroffice export fit file into a python dictionary

Functions:

```python
parse_bnmroffice_fit(filename)
```

### draw_1f_superimpose

Draw 1f runs, superimposed. Options for shifting by baseline or window center for comparison.

Functions:

```python
def draw(run,year,rebin=1,freq_shift=True,base_shift=True,label='run'):
    """
        runlist:    list of runs to draw or int
        yearlist:   list of years corresponding to run, or int
        rebin:      rebin factor, list or int
        freq_shift: if true, shift frequencies to ppm, fixed to window center
        base_shift: if true, baseline shift to zero
        label:      bdata attribute to set to label
    """
```

### get_experimenter_names

Find all author names for a given experiment

Call from the command line as an executable.

```bash
get_experimenter_names.py exp_number
```

Prints as output a list of experimenters for each year and the number of runs in which they participated. 

### campvarlog

Object for reading campvarlog files. Constructor

```
campvarlog(filename)
```

Methods:

* `draw(self,ycolumn,xcolumn='time')`

### read_striptool

Function

```
read_striptool(filename)
```

Returns pandas dataframe with index as datetime objects. 

### rga

Object for reading rga output files. 

Constructor: 

```
rga(filename)
```

Takes either the histogram or PvT file types. 

Methods

* [`draw(ax = None)`](https://github.com/dfujim/bnmr_tools/blob/f27bfc68590d3f1a594cdfce7dadc35265acf257/rga.py#L145)

### srim

Object for reading SRIM RANGE.txt and TDATA.txt files

Constructor

```
srim(fetchdir='.', range_txt='RANGE.txt'))
```

Methods

* `get_range(filename)`
* `interp_hist(x)`
* `draw()`
