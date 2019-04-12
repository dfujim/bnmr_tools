# Misc BNMR Python Tools

### `beta_averaging`

Code originally written by rmlm for T1 average with uncorrelated errors. 

Functions: 

```python
t1avg(T1, beta)                       # t1 avg
dt1avg(T1, dT1, beta, dbeta)          # error
```

### `file_checkup`

File inspector for start of beam. Make sure variables are logged properly. 

Functions: 

```python

file_checkup(area='bnmr',run=-1,year=-1)
```

As a command line tool: 

```
python3 file_checkup [area] [run] [year]
```

### `parse_bnmroffice_fit`

Turn bnmroffice export fit file into a python dictionary

Functions: 

```python 
parse_bnmroffice_fit(filename)
```

### `draw_1f_superimpose`

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