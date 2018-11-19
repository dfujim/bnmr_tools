# Fit depth scan with a function, convoluted with the implantation profile. 
# Derek Fujimoto
# Nov 2018

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.integrate import quad

# =========================================================================== # 
class fitresult(object):
    """
        Store fit results and draw. 
    """
    
    # ======================================================================= #
    def __init__(self,par,cov,fn,x=None,y=None,dy=None):
        
        # Set values
        self.par = par
        self.cov = cov
        self.std = np.diag(cov)**0.5
        self.fn = fn
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.dy = np.asarray(dy)
    
    # ======================================================================= #
    def draw(self,**plotargs):
        fitx = np.arange(500)/500*(max(self.x)-min(self.x))+min(self.x)
        plt.plot(fitx,self.fn(fitx,*self.par),**plotargs)
    
    # ======================================================================= #
    def draw_res(self,norm=False,**errorbarargs):
        """If norm == True, draw standardized residuals"""
        
        # get data
        res = self.get_residuals()
        
        if norm: 
            tag = self.dy != 0
            
            res = res[tag]
            res /= self.dy[tag]
            x = self.x[tag]
            dy = np.ones(len(res))
        else:
            dy = self.dy
            x = self.x
            
        # draw
        plt.errorbar(x,res,dy,**errorbarargs)
        plt.axhline(0,ls='-',color='k')
            
    # ======================================================================= #
    def get_chi(self):
        """Get Chisquared/DOF"""
        
        # check for data
        if self.x == None or self.y == None:
            raise RuntimeError('x or y is not data set.\nx = %s\ny = %s' % \
                (str(type(self.x)),str(type(self.y))))
        x = self.x
        y = self.y
        
        # check for errors
        if self.dy != None: 
            dy = self.dy
        else:
            print('Warning: chisquared has units due to dy not input.')
            dy = np.ones(len(self.x))
        
        # remove zeros
        tag = [self.dy!=0]
        x = x[tag]
        y = y[tag]
        dy = dy[tag]
        
        
        # get number of parameters
        npar = len(self.par)
        
        # get chisquared
        self.chi = np.sum(np.square((y-self.fn(x,*self.par)/dy)))/npar
        
        return self.chi

    # ======================================================================= #
    def get_cov(self):  return self.cov
    def get_data(self): return (self.x,self.y,self.dy)
    def get_fn(self):   return self.fn
    def get_par(self):  return self.par
    def get_std(self):  return self.std

    # ======================================================================= #
    def get_residuals(self):
        """Data-model"""
        self.res = self.y-self.fn(self.x,*self.par)
        return self.res

# =========================================================================== #
def gaussian(x,mean,std):
    """Truncated gaussian""" 
    return np.exp(-0.5*np.square((x-mean)/std))*2/\
        ( (1+erf(mean/(np.sqrt(2)*std))) * np.sqrt(np.pi*2)*std ) 

# =========================================================================== #
def depth_fit(E,y,dy,fn,impl_mean_fn,impl_strag_fn,**fitargs):
    """
        Fit data convoluting truncated gaussian depth profile with input 
        function.
        
        E:              implantation energy
        y:              quantity measured as a fn of energy
        dy:             error in y
        fn:             function handle to fit to y as a function of E
        impl_mean_fn:   mean implantation depth as a function of E
        impl_strag_fn:  straggle (stdev) of implantation depth profile as a fn 
                        of E
        fitargs:        parameters to pass to curve_fit
        
        returns: fitresult objet
    """
    
    # make fitting function 
    def fitfn(energy,*par):
        out = []
        for E in energy: 
            integrand = lambda x : fn(x,*par)*gaussian(x,impl_mean_fn(E),impl_strag_fn(E))
            out.append(quad(integrand,0,np.inf)[0])
        return np.array(out)

    # do the fit
    par,cov = curve_fit(fitfn,E,y,sigma=dy,**fitargs)
    
    # make output
    return fitresult(par,cov,fn,E,y,dy)
    
    
    
