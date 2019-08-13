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
from skimage.transform import hough_circle, hough_circle_peaks,rescale
from matplotlib.patches import Circle,Ellipse

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
        
        data = rescale(data,resc,order=3,multichannel=False,
                           preserve_range=True) 
        
    return data

def draw(filename,blacklevel=0,alpha=1,cmap='Greys',rescale_pixels=True,**kwargs):
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
    
    # draw
    plt.imshow(data,alpha=alpha,cmap=cmap+'_r',**show_options)

def draw_edges(filename,blacklevel=0,sigma=1,alpha=1,cmap='Greys',
               rescale_pixels=True,draw_image=True,**kwargs):
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
    
    # get edges
    edges = canny(data,sigma=sigma,low_threshold=0, high_threshold=1)
    
    # draw
    if draw_image:
        plt.imshow(data,alpha=1,cmap='Greys_r',**show_options)
        edges = np.ma.masked_where(~edges,edges.astype(int))
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
    
def draw_contour(filename,ncontours=5,blacklevel=0,alpha=1,cmap='Greys',rescale_pixels=True,**kwargs):
    """
        Draw contours of fits file to matplotlib figure
        
        filename:   name of fits file to read
        ncontours:  number of contours to draw
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
    ax.contour(X,Y,data,levels=ncontours,cmap=cmap+'_r',**show_options)
    
def detect_circles(filename,rad_range,ncircles=1,sigma=1,blacklevel=0,
                   draw=False,rescale_pixels=True,**kwargs):
    """
        Detect circles in image
        
        filename:   name of fits file to read
        rad_range:  specify raidus search range (lo,hi)
        blacklevel: value to set to black, all pixels of lower value raised 
                    to this level
        alpha:      draw transparency
        cmap:       colormap
        
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
                                                total_num_peaks=ncircles)
    
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
    
def get_center(filename,blacklevel=0,draw=False,rescale_pixels=True,**kwargs):
    """
        Get image center of mass
        
        filename:   name of fits file to read
        radii:      specify raidus ranges (lo,hi)
        blacklevel: value to set to black, all pixels of lower value raised 
                    to this level
    """
    
    # get raw data
    data = get_data(filename,blacklevel=blacklevel,rescale_pixels=rescale_pixels)
    
    # compress
    sumx = np.sum(data,axis=0)
    sumy = np.sum(data,axis=1)
    
    # fit with gaussian
    gaus = lambda x,x0,sig,amp,base : amp*np.exp(-((x-x0)/sig)**2)+base
    parx,cov = curve_fit(gaus,np.arange(len(sumx)),sumx,p0=(180,10,10,sumx[0]))
    stdx = np.diag(cov)**0.5
    
    pary,cov = curve_fit(gaus,np.arange(len(sumy)),sumy,p0=(260,10,10,sumy[0]))
    stdy = np.diag(cov)**0.5               
    
    # draw
    if draw:
        plt.figure()
        plt.plot(sumx/parx[3],label='x')
        plt.plot(sumy/pary[3],label='y')
        
        fitx = np.linspace(0,max(len(sumx),len(sumy)),5000)
        plt.plot(fitx,gaus(fitx,*parx)/parx[3],color='k')
        plt.plot(fitx,gaus(fitx,*pary)/pary[3],color='k')     
        plt.legend()
        
        plt.figure()
        plt.imshow(data,cmap='Greys_r',**show_options)
        plt.plot(parx[0],pary[0],'x')
            
    # return 
    return (parx[0],pary[0],parx[1],pary[1])

def gaussian2D(x,y,x0,y0,sigmax,sigmay,amp,offset):
    """Gaussian in 2D"""
    return amp*np.exp(-((x-x0)**2/(2*sigmax**2)-(y-y0)**2/(2*sigmay**2)))+offset
    
def fit(filename,function,blacklevel=0,rescale_pixels=True,**fitargs):
    """
        Fit function to fits file
    """
    
    # get data
    data = get_data(filename,blacklevel=blacklevel,rescale_pixels=rescale_pixels)
    
    # flatten the image
    shape = data.shape
    flat = np.concatenate(data)
    
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
                        facecolor='none',linewidth=1)
        ax.add_patch(circle)
    
    
    
