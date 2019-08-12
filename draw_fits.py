# draw fits file
# Derek Fujimoto
# July 2019

import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import os
import numpy as np
from scipy.optimize import curve_fit
from skimage import filters
from skimage.draw import circle_perimeter
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from matplotlib.patches import Circle

def get_data(filename,blacklevel=0,filterlevel=0,):
    """
        Get xy data from fits file. Values are brightness of pixel. 
        
        filename:   name of file to open, or list of files
        blacklevel: value of a "black" pixel, everything below this is 
                    considered oversaturated
        filterlevel:value to set to black, for fitting purposes
        
        Output:     2D array of values, or list of 2D arrays
    """
    if type(filename) is str:
        filename = os.path.join(os.getcwd(),filename)
        fid = fits.open(filename)[0]
        dat = fid.data
    else:
        fname = filename[0]
        fid = fits.open(fname)[0]
        dat = fid.data
        
        for fname in filename[1:]:
            fid = fits.open(fname)[0]
            dat += fid.data
        
    # fix bad pixels: set to max
    dat[dat<blacklevel] = np.max(dat)
    
    # clean: remove lowest values
    if filterlevel:
        dat[dat<filterlevel] = np.min(dat[dat!=0])
    
    return dat

def draw(filename,blacklevel=0,filterlevel=0,alpha=1,cmap='Greys',**kwargs):
    """
        Draw fits file to matplotlib figure
        
        filename:   name of fits file to read
        blacklevel: value of a "black" pixel, everything below this is 
                    considered oversaturated
        filterlevel:value to set to black, for fitting purposes
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
    data = get_data(filename,blacklevel=blacklevel,filterlevel=filterlevel)
    
    # draw
    plt.imshow(data,alpha=alpha,origin='lower',interpolation='nearest',
               cmap=cmap+'_r')

def draw_edges(filename,blacklevel=0,filterlevel=0,sigma=1,alpha=1,cmap='Greys',**kwargs):
    """
        Draw fits file to matplotlib figure
        
        filename:   name of fits file to read
        blacklevel: value of a "black" pixel, everything below this is 
                    considered oversaturated
        filterlevel:value to set to black, for fitting purposes
        sigma:      Standard deviation of the Gaussian filter.
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
    data = get_data(filename,blacklevel=blacklevel,filterlevel=filterlevel)
    
    # get edges
    edges = canny(data,sigma=sigma, low_threshold=0, high_threshold=50)
    
    # draw
    plt.imshow(edges.astype(int),alpha=alpha,origin='lower',interpolation='nearest',
               cmap=cmap)

def draw_sobel(filename,blacklevel=0,filterlevel=0,alpha=1,cmap='Greys',**kwargs):
    """
        Draw fits file to matplotlib figure
        
        filename:   name of fits file to read
        blacklevel: value of a "black" pixel, everything below this is 
                    considered oversaturated
        filterlevel:value to set to black, for fitting purposes
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
    data = get_data(filename,blacklevel=blacklevel,filterlevel=filterlevel)
    
    # draw
    plt.imshow(filters.sobel(data),alpha=alpha,origin='lower',
                interpolation='nearest',cmap=cmap)
    
def draw_contour(filename,blacklevel=0,ncontours=5,filterlevel=0,alpha=1,cmap='Greys',**kwargs):
    """
        Draw contours of fits file to matplotlib figure
        
        filename:   name of fits file to read
        blacklevel: value of a "black" pixel, everything below this is 
                    considered oversaturated
        ncontours:  number of contours to draw
        filterlevel:value to set to black, for fitting purposes
        alpha:      draw transparency
        cmap:       colormap    
    """
    
    # get raw data
    data = get_data(filename)
    
    # fix bad pixels: set to max
    data[data<blacklevel] = np.max(data)
    
    # clean: remove lowest values
    if filterlevel:
        data[data<filterlevel] = np.min(data[data!=0])
    
    # draw
    X,Y = np.meshgrid(*tuple(map(np.arange,data.shape[::-1])))
    ax = plt.gca()
    ax.contour(X,Y,data,levels=ncontours,cmap=cmap+'_r',origin='lower')
    
def detect_circles(filename,radii,ncircles=1,sigma=1,blacklevel=0,filterlevel=0,
                   draw=False,**kwargs):
    """
        Detect circles in image
        
        filename:   name of fits file to read
        radii:      specify raidus ranges (lo,hi)
        blacklevel: value of a "black" pixel, everything below this is 
                    considered oversaturated
        filterlevel:value to set to black, for fitting purposes
        alpha:      draw transparency
        cmap:       colormap
    """
    
    # get raw data
    data = get_data(filename,blacklevel=blacklevel,filterlevel=filterlevel)
    
    # get edges
    edges = canny(data,sigma=sigma, low_threshold=0, high_threshold=1)
    
    # get radii
    hough_radii = np.arange(*radii, 2)
    hough_res = hough_circle(edges, hough_radii)
    
    # select cicles 
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                total_num_peaks=ncircles)
    
    # draw
    if draw:
        plt.imshow(edges.astype(int),origin='lower',interpolation='nearest',
                   cmap='Greys')
        plt.imshow(data,origin='lower',interpolation='nearest',
                   cmap='Greens_r',alpha=0.5)
        for center_y, center_x, radius in zip(cy, cx, radii):
            circle = Circle((center_x,center_y),radius,
                        facecolor='none',linewidth=1,edgecolor='r')
            plt.gca().add_patch(circle)
            
    # return 
    return (cx,cy,radii)
    
def get_center(filename,blacklevel=0,filterlevel=0,draw=False,**kwargs):
    """
        Get image center of mass
        
        filename:   name of fits file to read
        radii:      specify raidus ranges (lo,hi)
        blacklevel: value of a "black" pixel, everything below this is 
                    considered oversaturated
        filterlevel:value to set to black, for fitting purposes
    """
    
    # get raw data
    data = get_data(filename,blacklevel=blacklevel,filterlevel=filterlevel)
    
    # invert
    data2 = np.copy(data)
    data2-= np.max(data2)
    data2 = np.abs(data2)
    
    # normalize
    data2 = data2/np.min(data2[data2!=0])
    
    # compress
    sumx = np.sum(data2,axis=0)
    sumy = np.sum(data2,axis=1)
    
    # fit with gaussian
    gaus = lambda x,x0,sig,amp : amp*np.exp(-((x-x0)/sig)**2)
    parx,cov = curve_fit(gaus,np.arange(len(sumx)),sumx,p0=(180,10,10))
    stdx = np.diag(cov)**0.5
    
    pary,cov = curve_fit(gaus,np.arange(len(sumy)),sumy,p0=(200,10,10))
    stdy = np.diag(cov)**0.5
    
    # draw
    if draw:
        plt.figure()
        plt.plot(sumx,label='x')
        plt.plot(sumy,label='y')
        
        fitx = np.linspace(0,max(len(sumx),len(sumy)),5000)
        plt.plot(fitx,gaus(fitx,*parx),color='k')
        plt.plot(fitx,gaus(fitx,*pary),color='k')
        plt.legend()
        
        plt.figure()
        plt.imshow(data,origin='lower',interpolation='nearest',
                   cmap='Greys_r')
        plt.plot(parx[0],pary[0],'x')
            
    # return 
    return (parx[0],pary[0],parx[1],pary[1])

def gaussian2D(x,y,x0,y0,sigmax,sigmay,amp,offset):
    """Gaussian in 2D"""
    return amp*np.exp(-((x-x0)**2/(2*sigmax**2)+(y-y0)**2/(2*sigmay**2)))+offset
    
def fit(filename,function,blacklevel=0,filterlevel=0,**fitargs):
    """
        Fit function to fits file
    """
    
    # get data
    data = get_data(filename,blacklevel=blacklevel,filterlevel=filterlevel)
    
    # flatten the image
    shape = dat.shape
    flat = np.concatenate(dat)
    
    # get number of fit parameters (first two are x,y)
    npar = len(function.__code__.co_varnames)-2
    if 'p0' not in fitargs:
        fitargs['p0'] = np.ones(npar)
        
    # get zero
    zero = np.min(flat)
    flat -= zero
    
    # flatten the funtion 
    def fitfn(xy,*pars):
        # get pixel indexes
        x = np.arange(shape[0])
        y = np.arange(shape[1])
        
        # return flattened output
        xarr,yarr = np.meshgrid(x,y)
        output = [function(ix,iy,*pars) for ix,iy in zip(xarr,yarr)]
        return np.concatenate(output)
    
    # fit
    return curve_fit(fitfn,np.arange(len(flat)),flat,**fitargs)
    
def draw_2dfit(x0,y0,sigmax,sigmay,*par):
    """Draw the fit function on the image as contours"""
    
    # draw the center
    plotted = plt.plot(x0,y0,'x')
    color = plotted[0].get_color()
    
    # draw contours
    ax = plt.gca()
    
    for i in range(1,4):
        circle = Ellipse((x0,y0),sigmay*i,sigmax*i,edgecolor=color,
                        facecolor='none',linewidth=2)
        ax.add_patch(circle)
    
    
    
