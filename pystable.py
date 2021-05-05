import math
from math import pi,tan,atan,sin,cos,exp
import random
from pygraph.classes.digraph import digraph
from pygraph.algorithms.minmax import shortest_path
from scipy.optimize import fmin
from scipy.integrate import quad
import numpy as np
import scipy as sp

# dstab, pstab, qstab

# abstract parameterisation data type
class ParnType:
    def __init__(self, name):
        self._name = name
    _name = 'unnamed parameterisation'
    def __repr__(self):
        return '<Parameterisation: '+self._name+'>'

pn_ST = ParnType('pn_ST')
pn_ZolA = ParnType('pn_ZolA')
pn_ZolM = ParnType('pn_ZolM')
pn_alpharho = ParnType('pn_alpharho')
pn_CMS = ParnType('pn_CMS')

# overengineered much?
_pnrels = digraph()
_pnrels.add_nodes([pn_ST,pn_ZolA,pn_ZolM,pn_alpharho,pn_CMS])
_pnrels.add_edge((pn_ZolA,pn_ST), attrs=[
    ('conv',
        lambda oldps:
        dict(alpha=oldps['alpha'], beta=oldps['beta'],
        mu=oldps['lamb']*oldps['gamma'],
        sigma=math.pow(oldps['lamb'],1/float(oldps['alpha'])))
    )])
def _ST_to_ZolA(oldps):
    tmplamb = math.pow(oldps['sigma'],oldps['alpha'])
    return dict(alpha=oldps['alpha'], beta=oldps['beta'],
        lamb=tmplamb, gamma=oldps['mu']/float(tmplamb))
_pnrels.add_edge((pn_ST,pn_ZolA), attrs=[('conv',_ST_to_ZolA)])
def _ZolM_to_ZolA(oldps):
    newgamma = oldps['gamma'] - oldps['beta']*math.tan(math.pi*oldps['alpha']*0.5)
    return dict(alpha=oldps['alpha'], beta=oldps['beta'],
        lamb=oldps['lamb'], gamma=newgamma)
_pnrels.add_edge((pn_ZolM,pn_ZolA), attrs=[('conv',_ZolM_to_ZolA)])
def _ZolA_to_ZolM(oldps):
    newgamma = oldps['gamma'] + oldps['beta']*math.tan(math.pi*oldps['alpha']*0.5)
    return dict(alpha=oldps['alpha'], beta=oldps['beta'],
        lamb=oldps['lamb'], gamma=newgamma)
_pnrels.add_edge((pn_ZolA,pn_ZolM), attrs=[('conv',_ZolA_to_ZolM)])
def _ZolM_to_CMS(oldps):
    if oldps['lamb'] != 1:
        raise ValueError('Tried to convert from ZolM to CMS, but '+
            'this is only possible when lamb=1, and here '+
            'lamb={0}.'.format(oldps['lamb']))
    if oldps['gamma'] != 0:
        raise ValueError('Tried to convert from ZolM to CMS, but '+
            'this is only possible when gamma=0, and here '+
            'gamma={0}.'.format(oldps['gamma']))
    return dict(alpha=oldps['alpha'],betaprime=oldps['beta'])
_pnrels.add_edge((pn_ZolM,pn_CMS), attrs=[('conv',_ZolM_to_CMS)])
_pnrels.add_edge((pn_CMS,pn_ZolM), attrs=[
    ('conv', lambda oldpns:
        dict(alpha=oldpns['alpha'], beta=oldpns['betaprime'],
        lamb=1, gamma=0)
    )])
def _alpharho_to_ST(oldps):
    tmpbeta = math.tan(math.pi*oldps['alpha']*(oldps['rho']-0.5))/ math.tan(math.pi*oldps['alpha']*0.5)
    c = math.sin(math.pi*oldps['alpha']*0.5)*math.cos(math.pi*oldps['alpha']*(oldps['rho']-0.5))
    return dict(alpha=oldps['alpha'],
        beta=tmpbeta,
        sigma=math.pow(c,1/float(oldps['alpha'])),
        mu=0)
_pnrels.add_edge((pn_alpharho,pn_ST), attrs=[('conv',_alpharho_to_ST)])
def _ST_to_alpharho(oldps):
    raise ValueError("I can't reliably do this conversion.")
_pnrels.add_edge((pn_ST,pn_alpharho), wt=100, attrs=[('conv',_ST_to_alpharho)]) # weight against this

class StableVar:
    def __init__(self,pn,**params):
        self.pn = pn
        if 'alpha' not in params:
            raise ValueError('Must pass parameter alpha.')
        if params['alpha'] <= 0 or params['alpha'] >= 2:
            raise ValueError('alpha must be strictly between 0 and 2')
        if pn == pn_ST:
            if 'beta' in params and (params['beta'] < -1 or params['beta'] > 1):
                raise ValueError('beta must be between -1 and 1')
            if 'sigma' in params and (params['sigma'] <= 0):
                raise ValueError('sigma must be strictly positive')
            params.setdefault('beta',0)
            params.setdefault('mu',0)
            params.setdefault('sigma',1)
            self.params = dict()
            for k in set(params.keys()).intersection({'alpha','beta','sigma','mu'}):
                self.params[k] = params[k]
        elif pn == pn_ZolA or pn == pn_ZolM: # param names/bounds are the same
            if 'beta' in params and (params['beta'] < -1 or params['beta'] > 1):
                raise ValueError('beta must be between -1 and 1')
            if 'lamb' in params and params['lamb'] <= 0:
                raise ValueError('lamb must be strictly positive')
            params.setdefault('beta',0)
            params.setdefault('gamma',0)
            params.setdefault('lamb',1)
            self.params = dict()
            for k in set(params.keys()).intersection({'alpha','beta','gamma','lamb'}):
                self.params[k] = params[k]
        elif pn == pn_CMS:
            if 'betaprime' in params and (params['betaprime'] < -1 or params['betaprime'] > 1):
                raise ValueError('betaprime must be between -1 and 1')
            params.setdefault('betaprime',0)
            self.params = dict()
            for k in set(params.keys()).intersection({'alpha','betaprime'}):
                self.params[k] = params[k]
        elif pn == pn_alpharho:
            params.setdefault('rho',0.5)
            if params['alpha'] < 1 and (params['rho'] < 0 or params['rho'] > 1):
                raise ValueError("When alpha < 1, rho must be in [0,1]; I have rho={0}.".format(params['rho']))
            if params['alpha'] > 1 and (params['rho'] < 1-1/float(params['alpha']) or params['rho'] > 1/float(params['alpha'])):
                raise ValueError("When alpha > 1, rho must be in [1-1/alpha,1/alpha]; I have rho = {0}, not in [{1},{2}]".format(params['rho'],1-1/float(params['alpha']),1/float(params['alpha'])))
            if params['alpha'] == 1 and params['rho'] != 0.5: # oh dear FIXME
                raise ValueError("When alpha = 1, rho must be 1/2. I have rho={0}.".format(params['rho']))
            self.params = dict()
            for k in set(params.keys()).intersection({'alpha','rho'}):
                self.params[k] = params[k]
        # elif...
        else:
            raise ValueError('Unknown parameterisation.')
    pn = pn_ST
    def get_rho(self):
        if self.pn == pn_ST:
            if self.params['alpha'] == 1 and (self.params['beta'] != 0 or self.params['mu'] != 0):
                raise ValueError('When alpha = 1, rho can only be calculated if the variable is symmetric(!)')
                # (is this the right exception?)
            elif self.params['alpha'] != 1 and self.params['mu'] != 0:
                raise ValueError('Can only calculate rho for a strictly stable variable, and this is not: alpha != 1 and mu != 0.')
            else:
                return 0.5 + math.atan(self.params['beta']*math.tan(math.pi*self.params['alpha']*0.5))/(math.pi*self.params['alpha'])
        else:
            return self.convert(pn_ST).get_rho() # lazy
    def convert(self,newpn):
        # OK shit, this fails if there is a conversion path
        # from self.pn to newpn (which we need) but not backwards
        # (which we don't)... can we traverse the spanning
        # tree in the other direction and fix this?
        pns_stree = shortest_path(_pnrels, newpn)[0]
        if self.pn not in pns_stree:
            raise ValueError('This conversion is not implemented.')
        cur_params = self.params
        old_pn = self.pn
        new_pn = pns_stree[old_pn]
        while new_pn != None:
            cur_params = dict(_pnrels.edge_attributes((old_pn,new_pn)))['conv'](cur_params)
            old_pn = new_pn
            new_pn = pns_stree[old_pn]
        return StableVar(old_pn, **cur_params)
    def convert_to_nearest(self, pns_list):
        pass # find nearest pn and convert
            
## non-standard functions from CMS.

# This cute algorithm is from section 1.14.1 in
# Higham, Accuracy and stability of numerical algorithms.
# (found via http://www.plunk.org/~hatch/rightway.php)
def _D2(x):
    u = math.exp(x)
    if u == 1:
        return x # x == 0
    return (u-1)/math.log(u)

# test suggested by http://www.plunk.org/~hatch/rightway.php
def _tan2(x):
    if 1 + x*x == 1:
        return 1
    else:
        return math.tan(x)/x

# kludgey but is there a better way...?
_tan2v = np.vectorize(_tan2)
_D2v = np.vectorize(_D2)

# return numpy array
def rstable(n,stvar):
    x = np.zeros(n)
    if stvar.pn == pn_CMS:
        return rstableCMS(n,stvar.params['alpha'],stvar.params['betaprime'])
    if stvar.pn == pn_ST:
        params = stvar.params
        alpha = params['alpha']
        beta = params['beta']
        mu = params['mu']
        sigma = params['sigma']
        if alpha == 1:
            shift = 0
        else:
            shift = beta*sp.tan(math.pi*alpha/2)
        return sigma*(rstableCMS(n,alpha,beta)+shift)+mu
        # double-check that multiplication by sigma is right :)
    #elif...
    else:
        return rstable(n,stvar.convert(pn_ST))

# return numpy array of n realisations from S^\prime(alpha,betaprime)
# as in [CMS]
def rstableCMS(n, alpha, betaprime):
    epsilon = 1-alpha
    k = 1 - abs(1-alpha)

    # some intermediate expressions are in terms of the parameter
    # \beta from CMS.
    if alpha == 1:
        beta = betaprime
    else:
        beta = 2/(math.pi*k)*math.atan(betaprime*math.tan(math.pi*epsilon/2))

    Phi0 = -0.5*math.pi*beta*k/alpha

    # these are the random inputs
    W = sp.random.exponential(1,n)
    Phi = sp.random.uniform(-0.5*math.pi,0.5*math.pi,n)

    # these are the auxiliary expressions defined in CMS
    if (epsilon < -0.99):
        tau = 0.5*math.pi*betaprime*epsilon*(1-epsilon)*_tan2v(0.5*math.pi*(1-epsilon))
    else:
        tau = betaprime/(0.5*math.pi*_tan2(0.5*math.pi*epsilon))
    a = 0.5*Phi*_tan2v(0.5*Phi)
    B = _tan2v(0.5*epsilon*Phi)
    b = 0.5*epsilon*Phi*B
    z = (1+pow(a,2))*(1-pow(b,2)+tau*Phi*B)/(W*(1-pow(a,2))*(1+pow(b,2)))
    # note ambiguity in CMS for the definition of d,
    # this is the correct one
    d = np.log(z)*_D2v(epsilon*np.log(z)/(1-epsilon))/(1-epsilon)

    # this is S^\prime from CMS
    Sp = (2*(a-b)*(1+a*b)-Phi*tau*B*(b*(1-pow(a,2))-2*a))/((1-pow(a,2))*(1+pow(b,2)))*(1+epsilon*d) + tau*d

    # ST now says that Sp ~ S_\alpha(1,\betaprime,-\betaprime\tan(\pi\alpha/2))
    # which is what we call the `CMS' parameterisation.

    return Sp

### HIGHLY EXPERIMENTAL

def sgn(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    else:
        return x
# np.sign does this as a ufunc

# here are some functions from Nolan (1999).
# he is using ZolM here.
# all *_Nol functions are from Nolan (1999).
_margin = 1e-14
_tol = 1e-15 # how do you choose this..?
def _theta0_Nol(alpha, beta):
    if alpha != 1:
        return 1/float(alpha)*atan(beta*tan(pi*alpha/2))
    else:
        return pi/2
def _V_Nol(theta, alpha, beta):
    if theta < _margin-_theta0_Nol(alpha,beta) or theta > pi/2-_margin:
        raise ValueError('theta={0} out of domain ({1},{2}) of V'.format(theta,_margin-_theta0_Nol(alpha,beta),pi/2-_margin))
    t0 = _theta0_Nol(alpha,beta)
    if alpha != 1:
        try:
            return pow(cos(alpha*t0),1/(float(alpha)-1))\
                *pow(cos(theta)/sin(alpha*(t0+theta)),\
                    alpha/(float(alpha)-1))\
                *cos(alpha*t0+(alpha-1)*theta)/cos(theta)
        except OverflowError: # kludge, but large values of V are
        # killed by _f_integrand_Nol anyway...!
            return float('inf')
    elif alpha == 1 and abs(beta)>_tol:
        return 2/pi\
            *(pi/2+beta*theta)/cos(theta)\
            *exp(1/float(beta)*(pi/2+beta*theta)*tan(theta))
    else:
        raise ValueError('Tried to compute V, but alpha=1 and '+
            'beta={0} is too close to zero.'.format(beta))
def _zeta_Nol(alpha,beta):
    if alpha != 1:
        return -beta*tan(pi*alpha/2)
    else:
        return 0
def _c2_Nol(x,alpha,beta):
    if alpha != 1:
        return alpha/(pi*abs(alpha-1)*(x-_zeta_Nol(alpha,beta)))
    elif abs(beta)>_tol:
        return 1/(2*abs(float(beta)))
    else:
        raise ValueError('Tried to compute c2, but alpha=1 and beta'+
            '={0} is too close to zero.'.format(beta))
def _c3_Nol(alpha):
    if alpha != 1:
        return sgn(1-alpha)/pi
    else:
        return 1/pi
def _g_Nol(theta,x,alpha,beta):
    if theta < _margin-_theta0_Nol(alpha,beta) or theta > pi/2-_margin:
        return -1 # protect from fmin overstepping
    if alpha != 1:
        return pow(x-_zeta_Nol(alpha,beta),alpha/(float(alpha)-1))\
            *_V_Nol(theta,alpha,beta)
    elif abs(beta)>_tol:
        return exp(-pi*x/(2*float(beta)))*_V_Nol(theta,alpha,beta)
    else:
        raise ValueError('Tried to compute g, but alpha=1 and beta'+
            '={0} is too close to zero.'.format(beta))

# debatable where the minimisation should be started;
# pi/2-epsilon seems to work in most cases, whereas
# the average of theta0 and pi/2 often gets stuck
## we don't special-case alpha=1 here, as Nolan seems
## to indicate that the same approach works.
def _theta2_Nol(x,alpha,beta):
    t2 = fmin(lambda theta: -_g_Nol(theta,x,alpha,beta), pi/2-0.1,
        disp=0
    )[0]
    if t2 < -_theta0_Nol(alpha,beta) or t2 > pi/2:
        raise ValueError('Calculated value of theta2 is outside domain of g.') # what is the right exception?
    else:
        return t2
# this is the integrand in (4) of Nolan
def _f_integrand_Nol(theta,x,alpha,beta):
    gtmp = _g_Nol(theta,x,alpha,beta)
    # if x > 43, x*exp(-x) < 1e-17.
    if gtmp > 43: # True if gtmp is Inf
        return 0
    else:
        return gtmp*exp(-gtmp)

def _f_Nol(x,alpha,beta):
    zeta_here = _zeta_Nol(alpha,beta)
    if alpha != 1:
        if abs(zeta_here-x)<_tol:
            return math.gamma(1+1/float(alpha))\
                *cos(_theta0_Nol(alpha,beta))\
                /(pi*pow(1+pow(zeta_here,2),1/float(2*alpha)))
        elif x < zeta_here:
            return _f_Nol(-x,alpha,-beta)
    elif abs(beta)<_tol:
        return 1/(pi*(1+pow(x,2)))
    # else: alpha != 1 and x > zeta, or else alpha == 1 and beta != 0
    ## XXX check this again w/ Nolan
    intl = sp.integrate.quad(np.vectorize(lambda theta: _f_integrand_Nol(theta,x,alpha,beta)), -_theta0_Nol(alpha,beta), pi/2)
    # , points=[_theta2_Nol(x,alpha,beta)]) # hopefully scipy can work this out.
    return _c2_Nol(x,alpha,beta)*intl[0]
    # [1] is error bound

def dstab(x,stvar):
    if stvar.pn == pn_CMS:
        return _f_Nol(x,stvar.params['alpha'],stvar.params['betaprime'])
    #elif...
    else:
        return dstab(x,stvar.convert(pn_CMS))

if __name__ == "__main__":
    print ('s = StableVar(pn_ST, alpha=0.5,beta=-0.5,mu=1,sigma=2)')
    s = StableVar(pn_ST, alpha=0.5,beta=-0.5,mu=1,sigma=2)
    x = rstable(10,s)

    s_rand = StableVar(pn_ST,
        alpha=random.uniform(0,2),
        beta=random.uniform(-1,1),
        mu=random.gauss(0,1),
        sigma=random.expovariate(1))
    s_rand_back_conv = s_rand.convert(pn_ZolM).convert(pn_ST)
    if s_rand_back_conv.pn != s_rand.pn:
        print ('Oh dear, the conversion has gone very wrong.')
    else:
        tol = 1e-14 # meh
        for param in s_rand.params:
            print ('Testing ',param,': ',)
            if abs(s_rand.params[param] - s_rand_back_conv.params[param]) < tol:
                print ('OK.')
            else:
                print ('Bad!',s_rand.params[param],'vs',s_rand_back_conv.params[param])

    s = StableVar(pn_CMS,alpha=1.01)
    t = np.linspace(-4,4,1000)
    dstabv = np.vectorize(lambda x: dstab(x,s))
