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

Funcitons: 

```python 
parse_bnmroffice_fit(filename)
```