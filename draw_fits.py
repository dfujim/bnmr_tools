# draw fits file
# Derek Fujimoto
# July 2019

import os
import numpy as np
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Ellipse

from astropy.io import fits

from scipy.optimize import curve_fit
from scipy.integrate import dblquad

import skimage as ski
from skimage import filters
from skimage.feature import canny
from skimage.transform import hough_circle,hough_circle_peaks
from skimage.transform import probabilistic_hough_line
from skimage.transform import rescale

show_options = {'origin':'lower',
                'interpolation':'nearest'}

def get_data(filename,blacklevel=0,rescale_pixels=True):
    """
        Get xy data from fits file. Values are brightness of pixel. 
        
        filename:       name of file to open
        blacklevel:     value to set to black, all pixels of lower value raised 
                        to this level
        rescale_pixels: if True, rescale image such that pixels are square
        
        Output:     2D array of values, or list of 2D arrays
    """
    filename = os.path.join(os.getcwd(),filename)
    fid = fits.open(filename)[0]
    data = fid.data

    # fix bad pixels: set to max
    data[data<fid.header['BZERO']] = np.max(data)
    
    # clean: remove lowest values
    if blacklevel:
        data[data<blacklevel] = blacklevel
    
    # rescale image to correct pixel size asymmetry
    if rescale_pixels:
        aspect = fid.header['YPIXSZ']/fid.header['XPIXSZ']
        
        # always enlarge image, never make it smaller
        if aspect > 1:      resc = (aspect,1)
        else:               resc = (1,1/aspect)
        
        data = rescale(data,resc,order=3,multichannel=False,preserve_range=True) 
    
    return data

def get_header(filename):
    """
        Get header info as a dictionary
    """
    filename = os.path.join(os.getcwd(),filename)
    fid = fits.open(filename)[0]
    return fid.header
    
def mask_data(data,mask=None):
    """
        Mask image data
        
        data:       2D np array 
        mask:       (x,y,r) specifying center and radius of circle to mask on
    """
    
    # masking
    if mask is not None: 
        window = np.ones(data.shape)
        rr,cc = ski.draw.circle(mask[1],mask[0],mask[2],shape=data.shape)
        window[rr,cc] = 0
        data = np.ma.array(data,mask=window)
    else:
        data = np.ma.asarray(data)
    
    return data
    
def draw(filename,blacklevel=0,alpha=1,cmap='Greys',rescale_pixels=True,mask=None,
         **kwargs):
    """
        Draw fits file to matplotlib figure
        
        filename:   name of fits file to read
        blacklevel: value to set to black, all pixels of lower value raised 
                    to this level
        alpha:      draw transparency
        cmap:       colormap
        
        Colormaps: 
            Greys
            Purples
            Yellows
            Blues
            Oranges
            Reds
            Greens
            ...
            
        https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    """
    
    # get raw data
    data = get_data(filename,blacklevel=blacklevel,rescale_pixels=rescale_pixels)
    data = mask_data(data,mask)
    
    # draw
    plt.imshow(data,alpha=alpha,cmap=cmap+'_r',**show_options)

def draw_edges(filename,blacklevel=0,sigma=1,alpha=1,cmap='Greys',
               rescale_pixels=True,draw_image=True,mask=None,**kwargs):
    """
        Draw fits file to matplotlib figure
        
        filename:   name of fits file to read
        blacklevel: value to set to black, all pixels of lower value raised 
                    to this level
        sigma:      Standard deviation of the Gaussian filter.
        alpha:      draw transparency
        cmap:       colormap
        draw_image: superimpose image 
    """
    
    # get raw data
    data = get_data(filename,blacklevel=blacklevel,rescale_pixels=rescale_pixels)
    data = mask_data(data,mask)
    
    # get edges
    data2 = np.copy(data)
    data2[data.mask] = blacklevel
    edges = canny(data2,sigma=sigma,low_threshold=0, high_threshold=1)
    
    # draw
    if draw_image:
        edges = np.ma.masked_where(~edges,edges.astype(int))
        plt.imshow(data,alpha=1,cmap='Greys_r',**show_options)
        plt.imshow(edges,alpha=1,cmap='Reds_r',**show_options)
    else:
        plt.imshow(edges.astype(int),alpha=alpha,cmap=cmap,**show_options)

def draw_sobel(filename,blacklevel=0,alpha=1,cmap='Greys',rescale_pixels=True,**kwargs):
    """
        Draw fits file to matplotlib figure
        
        filename:   name of fits file to read
        blacklevel: value to set to black, all pixels of lower value raised 
                    to this level
        alpha:      draw transparency
        cmap:       colormap
    """
    
    # get raw data
    data = get_data(filename,blacklevel=blacklevel,rescale_pixels=rescale_pixels)
    
    # draw
    plt.imshow(filters.sobel(data),alpha=alpha,cmap=cmap,**show_options)
    
def draw_contour(filename,n=5,blacklevel=0,alpha=1,cmap='Greys',rescale_pixels=True,**kwargs):
    """
        Draw contours of fits file to matplotlib figure
        
        filename:   name of fits file to read
        n:          number of contours to draw
        blacklevel: value to set to black, all pixels of lower value raised 
                    to this level
        alpha:      draw transparency
        cmap:       colormap    
    """
    
    # get raw data
    data = get_data(filename,blacklevel=blacklevel)
    
    # draw
    X,Y = np.meshgrid(*tuple(map(np.arange,data.shape[::-1])))
    ax = plt.gca()
    ax.contour(X,Y,data,levels=n,cmap=cmap+'_r',**show_options)
    
def detect_lines(filename,sigma=1,min_length=50,min_gap=3,theta=None,n=np.inf,
                 blacklevel=0,draw=True,rescale_pixels=True,**kwargs):
    """
        Detect lines in image
        
        filename:   name of fits file to read
        blacklevel: value to set to black, all pixels of lower value raised 
                    to this level
        n:          number of line s to find
        min_length: minimum length of lines to find
        min_gap:    minimum gap between pixels to avoid breaking the line    
        theta:      list of acceptable angles for the lines to point
        
        returns: list of points ((x0,y0),(x1,y1)) to identify the end points of 
                 the lines
    """
    
    # get raw data
    data = get_data(filename,blacklevel=blacklevel,rescale_pixels=rescale_pixels)
    
    # get edges
    edges = canny(data,sigma=sigma, low_threshold=0, high_threshold=1)
    
    # select lines
    lines = probabilistic_hough_line(edges,threshold=10,line_length=min_length,
                                     line_gap=min_gap,theta=theta)
    # draw
    if draw:
        plt.figure()
        plt.imshow(data,alpha=1,cmap='Greys_r',**show_options)
        edges = np.ma.masked_where(~edges,edges.astype(int))
        plt.imshow(edges,alpha=1,cmap='Reds_r',**show_options)
        
        for line in lines:
            plt.plot(*tuple(np.array(line).T))
            
    # return 
    return lines+2

def detect_hlines(filename,sigma=1,min_length=50,min_gap=3,n=np.inf,
                 blacklevel=0,draw=True,rescale_pixels=True,**kwargs):
    """
        Detect horizontal lines in image
        
        filename:   name of fits file to read
        blacklevel: value to set to black, all pixels of lower value raised 
                    to this level
        n:          number of line s to find
        min_length: minimum length of lines to find
        min_gap:    minimum gap between pixels to avoid breaking the line    
        
        returns: list of y positions to identify each line
    """
    
    # make a set of ranges about pi/2
    theta = np.linspace(np.pi/2-0.01,np.pi/2+0.01,30)
    
    # get lines 
    lines = detect_lines(filename=filename,sigma=sigma,min_length=min_length,
                         min_gap=min_gap,n=n,blacklevel=blacklevel,draw=draw,
                         rescale_pixels=rescale_pixels,theta=theta,**kwargs)
    
    # get y values of lines 
    return [l[0][1] for l in lines]
            
def detect_circles(filename,rad_range,n=1,sigma=1,blacklevel=0,
                   draw=True,rescale_pixels=True,**kwargs):
    """
        Detect circles in image
        
        filename:   name of fits file to read
        rad_range:  specify raidus search range (lo,hi)
        blacklevel: value to set to black, all pixels of lower value raised 
                    to this level
        alpha:      draw transparency
        cmap:       colormap
        n:          number of circles to find
        
        returns: (center_x,center_y,radius)
    """
    
    # get raw data
    data = get_data(filename,blacklevel=blacklevel,rescale_pixels=rescale_pixels)
    
    # get edges
    edges = canny(data,sigma=sigma, low_threshold=0, high_threshold=1)
    
    # get radii
    hough_radii = np.arange(*rad_range, 2)
    hough_res = hough_circle(edges, hough_radii)
    
    # select cicles 
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                total_num_peaks=n)
    
    # draw
    if draw:
        
        plt.imshow(data,alpha=1,cmap='Greys_r',**show_options)
        edges = np.ma.masked_where(~edges,edges.astype(int))
        plt.imshow(edges,alpha=1,cmap='Reds_r',**show_options)
        
        for center_y, center_x, radius in zip(cy, cx, radii):
            circle = Circle((center_x,center_y),radius,
                        facecolor='none',linewidth=1,edgecolor='g')
            plt.gca().add_patch(circle)
            
    # return 
    return (cx,cy,radii)
    
def get_center(filename,blacklevel=0,draw=True,rescale_pixels=True,mask=None,**kwargs):
    """
        Get image center of mass
        
        filename:   name of fits file to read
        radii:      specify raidus ranges (lo,hi)
        blacklevel: value to set to black, all pixels of lower value raised 
                    to this level
        mask:       (x,y,r) specifying center and radius of circle to mask on
    """
    
    # get raw data
    data = get_data(filename,blacklevel=blacklevel,rescale_pixels=rescale_pixels)
    fid = fits.open(filename)[0]
    black = max(blacklevel,fid.header['BZERO'])
    
    # mask
    data = mask_data(data,mask)
        
    # compress
    sumx = np.ma.mean(data,axis=0)
    sumy = np.ma.mean(data,axis=1)
    
    # shift baseline
    sumx -= black
    sumy -= black
    
    # normalize
    normx = np.ma.max(sumx)
    normy = np.ma.max(sumy)
    
    sumx /= normx
    sumy /= normy
    
    # fit with gaussian
    gaus = lambda x,x0,sig,amp,base : amp*np.exp(-((x-x0)/(2*sig))**2)+base
    
    parx,cov = curve_fit(gaus,np.arange(len(sumx)),sumx,p0=(180,10,1,0),
                            bounds=((0,0,0,-np.inf),np.inf))
    stdx = np.diag(cov)**0.5
    
    pary,cov = curve_fit(gaus,np.arange(len(sumy)),sumy,p0=(260,10,1,0),
                            bounds=((0,0,0,-np.inf),np.inf))
    stdy = np.diag(cov)**0.5               
    
    # draw
    if draw:
        plt.figure()
        plt.plot(sumx*normx,label='x')
        plt.plot(sumy*normy,label='y')
        
        fitx = np.linspace(0,max(len(sumx),len(sumy)),5000)
        plt.plot(fitx,gaus(fitx,*parx)*normx,color='k')
        plt.plot(fitx,gaus(fitx,*pary)*normy,color='k')     
        plt.legend()
        
        plt.figure()
        plt.imshow(data,cmap='Greys_r',**show_options)
        plt.errorbar(parx[0],pary[0],xerr=2*parx[1],yerr=2*pary[1],fmt='o',
                      fillstyle='none',markersize=9)
                      
        if pary[1] > 2 and parx[1] > 2:
            plt.ylim(pary[0]-pary[1]*6,pary[0]+pary[1]*6)   
            plt.xlim(parx[0]-parx[1]*6,parx[0]+parx[1]*6)
            
    # return 
    return (parx[0],pary[0],parx[1],pary[1])

def get_cm(filename,blacklevel=0,draw=True,rescale_pixels=True,mask=None,
           **kwargs):
    """
        Get image center of mass
        
        filename:   name of fits file to read
        radii:      specify raidus ranges (lo,hi)
        blacklevel: value to set to black, all pixels of lower value raised 
                    to this level
    """
    
    # get raw data
    data = get_data(filename,blacklevel=blacklevel,rescale_pixels=rescale_pixels)
    data = mask_data(data,mask)
    
    # compress
    sumx = np.ma.mean(data,axis=0)
    sumy = np.ma.mean(data,axis=1)
    
    # estimate center with weighted average
    sumx -= np.ma.min(sumx)
    sumy -= np.ma.min(sumy)
    
    nsumx = len(sumx)
    nsumy = len(sumy)
    
    cx = np.ma.average(np.arange(nsumx),weights=sumx)
    cy = np.ma.average(np.arange(nsumy),weights=sumy)

    # draw
    if draw:
        plt.figure()
        plt.imshow(data,cmap='Greys_r',**show_options)
        plt.plot(cx,cy,'x')
            
    # return 
    return (cx,cy)

def gaussian2D(x,y,x0,y0,sigmax,sigmay,amp,theta=0):
    """Gaussian in 2D - from wikipedia"""
    
    ct2 = np.cos(theta)**2
    st2 = np.sin(theta)**2
    s2t = np.sin(2*theta)
    
    sx = sigmax**2
    sy = sigmay**2
    
    a = 0.5*(ct2/sx + st2/sy)
    b = 0.25*s2t*(-1/sx + 1/sy)
    c = 0.5*(st2/sx + ct2/sy)
    
    return amp*np.exp(-(a*np.square(x-x0) + 2*b*(x-x0)*(y-y0) + c*np.square(y-y0)))

def fit2D(filename,function,blacklevel=0,rescale_pixels=True,**fitargs):
    """
        Fit general function to fits file
    """
    
    # get data
    data = get_data(filename,blacklevel=blacklevel,rescale_pixels=rescale_pixels)
    
    data = data[:300,:200]
    
    # flatten the image
    flat = np.ravel(data)
    
    # get number of fit parameters (first two are x,y)
    npar = len(function.__code__.co_varnames)-2
    if 'p0' not in fitargs:
        fitargs['p0'] = np.ones(npar)
        
    # get zero
    zero = np.min(flat)
    flat -= zero
    
    # normalize
    flat /= np.max(flat)
    
    # flatten the funtion 
    def fitfn(xy,*pars):    
        output = function(*xy,*pars)
        return np.ravel(output)
    
    # fit
    x = np.indices(data.shape)[::-1]
    return curve_fit(fitfn,x,flat,**fitargs)
    
def draw_2Dfit(shape,fn,*pars,levels=10,cmap='jet'):
    """Draw the fit function as contours"""
    
    # get function image
    x = np.arange(shape[1])    
    y = np.arange(shape[0])    
    gauss = np.zeros((len(y),len(x)))
    for i in y:
        gauss[i-y[0],:] = fn(x,i,*pars)

    # draw image
    X,Y = np.meshgrid(x,y)
    ax = plt.gca()
    CS = ax.contour(X,Y,gauss,levels=levels,cmap=cmap)
    return CS
    
def fit_gaussian2D(filename,blacklevel=0,rescale_pixels=True,
                   draw_output=True,**kwargs):
    """
        Fit 2D gaussian to image
    """
    
    # get data 
    data = get_data(filename,blacklevel=blacklevel,rescale_pixels=rescale_pixels)
    
    # estimate moments https://scipy-cookbook.readthedocs.io/items/FittingData.html
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum()) 
    
    # fit 
    p0 = (x,y,width_x,width_y,1,0)
    par,cov = fit2D(filename,gaussian2D,blacklevel=blacklevel,
                  rescale_pixels=rescale_pixels,p0=p0)
    std = np.diag(cov)**0.5
    
    # draw output
    if draw_output:
        plt.figure()    
        draw(filename,blacklevel=blacklevel,rescale_pixels=rescale_pixels)
        draw_2Dfit(data.shape,gaussian2D,*par,**kwargs)
    
    return(par,std)
    
def get_gaussian2D_overlap(ylo,yhi,xlo,xhi,x0,y0,sx,sy,amp,theta=0):
    """
        Get integral of gaussian2D PDF within some interval, normalized to the 
        area such that the returned overlap is the event probability within the 
        range. 
        
        ylo:    lower integration bound [outer] (float)
        yhi:    upper integration bound [outer] (float)
        xlo:    lower integration bound [inner] (lambda function)
        xlhi:   upper integration bound [inner] (lambda function)
        x0,y0:  gaussian mean location
        sx,sy:  standard deviation
        amp:    unused in favour of normalized amplitude (present for ease of use)
        theta:  angle of rotation
        
            integration is: 
                
                int_y int_x G(x,y) dx dy
        
        
        returns overlap as given by dblquad
    """
    
    # get normalized amplitude
    # https://en.wikipedia.org/wiki/Gaussian_function
    a = 0.5*(np.cos(theta)/sx)**2 + 0.5*(np.sin(theta)/sy)**2
    b = 0.25*-(np.sin(theta)/sx)**2 + 0.25*(np.sin(theta)/sy)**2
    c = 0.5*(np.sin(theta)/sx)**2 + 0.5*(np.cos(theta)/sy)**2
    amp = np.sqrt(a*c-b**2)/np.pi
    
    # make PDF
    gaus = lambda x,y: gaussian2D(x,y,x0,y0,sx,sy,amp,theta)
    
    # integrate: fraction of beam overlap
    return dblquad(gaus,ylo,yhi,xlo,xhi)[0]
    
    

