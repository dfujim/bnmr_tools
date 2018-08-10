# MODIFIED FROM rmlm
# Note: <1/T1> != 1/<T1>, see comments below Eq20 in 
# Lindsey, C. P. & Patterson, G. D. J. Chem. Phys. 73, 3348–1484 (1980).

from scipy.special import gamma, polygamma
from numpy import sqrt,square

# Calculate <T_1> (i.e., beta-averaged relaxation time)
def t1avg(T1, beta) :
    return T1*gamma(1./beta)/beta

# Calculate uncertainty in <T_1> (i.e., beta-averaged relaxation time)
def dt1avg(T1, dT1, beta, dbeta) :
    betai = 1./beta
    pd_T1 = gamma(1./beta)/beta
    #~ pd_beta = T1*( beta + polygamma(0, 1./beta) )/( beta*gamma(1./beta) )
    pd_beta = T1*gamma(betai)*(1+betai*polygamma(0,betai))*square(betai)
    return sqrt( (pd_T1*dT1)**2 + (pd_beta*dbeta)**2 )
