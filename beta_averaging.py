# MODIFIED FROM rmlm
# Note: <1/T1> != 1/<T1>, see comments below Eq20 in 
# Lindsey, C. P. & Patterson, G. D. J. Chem. Phys. 73, 3348–1484 (1980).

from scipy.special import gamma, polygamma
from numpy import sqrt,square

# Calculate <T_1> (i.e., beta-averaged relaxation time)
def t1avg(T1, beta) :
    return T1*gamma(1./beta)/beta

# Calculate uncertainty in <T_1> (i.e., beta-averaged relaxation time)
def dt1avg(T1, dT1, beta, dbeta, cov=0) :
    """cov = covarince between T1 and beta"""
    
    betai = 1./beta
    pd_T1 = gamma(1./beta)/beta
    pd_beta = -T1*gamma(betai)*(1+betai*polygamma(0,betai))*square(betai)
    return sqrt( (pd_T1*dT1)**2 + (pd_beta*dbeta)**2 + 2*cov*pd_T1*pd_beta) 

# Calculate uncertainty in <T_1> (i.e., beta-averaged relaxation time), from 1/T1
def dt1avg_fromL(lambda1, dlambda1, beta, dbeta, cov=0) :
    """cov = covarince between 1/T1 and beta"""
    
    # get covariance between T1 and beta, from 1/T1 and beta
    cov = -cov/square(lambda1)
    
    # get T1
    T1 = 1/lambda1
    dT1 = dlambda1/square(lambda1)
        
    return dt1avg(T1, dT1, beta, dbeta, cov) 
