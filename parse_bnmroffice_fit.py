# Fetch useful info from report.fit files produced by bnmroffice_v1
# Derek Fujimoto
# May 2017

def parse_bnmroffice_fit(filename):
    """Parse bnmroffice .fit file (i.e. report.fit) and return a dictionary with
    the data columns.
    
    Derek Fujimoto
    May 2017
    """
    
    # read file
    fid = open(filename,'r')
    lines = fid.read().split('\n')

    header = lines[0]
    lines = lines[1:]
    
    fid.close()
    
    # initialize data dictionary
    data = {}
    
    # set keys
    header = header.split()
    for h in header: 
        data[h] = []
    
    # get data
    for line in lines:
        for i,l in enumerate(line.split()):
            data[header[i]].append(float(l))

    # return data dictionary
    return data
    
