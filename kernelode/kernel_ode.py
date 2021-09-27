import os
import matplotlib.pyplot as plt

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.nn.module import PyroParam


from torchdiffeq import odeint
from math import pi



π = torch.tensor(pi)

def ode_kernel(X,Y, σ_b, σ_w, n=500.0,dim=1,method='euler', atol=0.01, rtol=1e-4, T=1, num_steps=100):
    #  Model: dk /dt = f(t, k )  where k is the kernel matrix   
    
    X = X.reshape(-1,dim)
    Y = Y.reshape(-1,dim)

    ts = torch.linspace(0, T, num_steps) 
    dt = T / n # Uniform scaling
    
    k_0 = lambda x,y,σ_b, σ_w: σ_b**2 + 0.5 * σ_w**2 * x.mm(y.T) # Initial gram matrix

    y0 = k_0(X,Y, σ_b, σ_w)

    def f(γ):
        ϵ = 1e-6
        γ = ϵ + (1 - 2 * ϵ) * torch.clip(γ,-1,1)
        
        return (1.0 / π) * (torch.sqrt(1.0 - γ**2) - γ * torch.arccos(γ))

    def k_xx_t_x(t,X_):
        k0 = (σ_b**2 + 0.5 * σ_w**2 * (X_**2).sum(axis=1) )
        result = torch.exp(0.5 * σ_w**2 * t) * k0 + 2 * (σ_b**2 / σ_w**2) * (torch.exp(0.5 * σ_w**2 * t) - 1.0)
        return result
   
    k_xx_t_ = lambda t: k_xx_t_x(t,X)
    k_yy_t_ = lambda t: k_xx_t_x(t,Y)
    
    def c_t(t,k):
        kxxt = k_xx_t_(t).reshape(-1,1)
        kyyt = k_yy_t_(t).reshape(-1,1)

        result = k / torch.sqrt(kxxt.mm(kyyt.T))
        return result

    def drift(t,k):
        ct = c_t(t,k)

        result = σ_b**2 + 0.5 * σ_w**2 * (1.0 + (f(ct) / ct)  ) * k 

        return result
    
    
    if method=='euler':
        out = odeint(drift, y0, ts, method="euler", options={"step_size":dt})
    elif method=='dopri5':
        out = odeint(drift, y0, ts, method='dopri5', atol=atol, rtol=rtol)
    else:
        out = odeint(drift, y0, ts, method=method)
    return out


class ODEKernel(gp.kernels.Kernel):
    """
    Returns a new kernel which acts like a product/tensor product of two kernels.
    The second kernel can be a constant.
    """
    def __init__(self, input_dim, σ_b=None, σ_w=None, active_dims=None,
                 name=None):
        super(ODEKernel, self).__init__(input_dim, active_dims)

        if σ_b is None:
            self.σ_b = torch.tensor(1.)
            
        if σ_w is None:
            self.σ_w = torch.tensor(1.)    
        
#         self.σ_b = torch.tensor(σ_b)
#         self.σ_w = torch.tensor(σ_w)

        # Comenting these two lines in favour for the two above
        # seem to fix most instability/jitter related issues
        self.σ_b = PyroParam(torch.tensor(σ_b), constraints.positive) 
        self.σ_w = PyroParam(torch.tensor(σ_w), constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if Z is None: Z = X
        return ode_kernel(X,Z, self.σ_b , self.σ_w, dim=self.input_dim)[-1,...]