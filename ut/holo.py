from importlib import reload
import torch as tr
import torch
import torch.fft as trf
import torch.distributions as trd
from torch.fft import fft as F
from torch.fft import ifft as iF
from torch.fft import fftn as Fn
from torch.fft import ifftn as iFn
coo = torch.sparse_coo_tensor

# def Amp(x): return (x.real**2 + x.imag**2)**0.5
# def Phase(x): return tr.atan(x.imag/x.real)

def A(x): return (x.real**2 + x.imag**2)**0.5
def P(x): return tr.atan(x.imag/x.real)


# def lF(A, dim =  (0,1)): return tr.log(Fn(A, dim = dim))
# def iFe(A, dim = (0,1)): return iFn(tr.exp(A), dim = dim)
# def Fe(A, dim =  (0,1)): return Fn(tr.exp(A), dim = dim)
#
# def Conv(A, B): return iFe(lF(A) + lF(B))
# def Convs(A): return iFn(tr.exp(tr.log(Fn(r, dim=A.shape[1:-1])).sum(dim=0)) , dim= A.shape[1:-1])
#
# def Corr(A, B): return iFe(lF(A) - lF(B))
# def Delta(A): return Corr(A, A)
# def Inv(A): return Corr(Delta(A), A)
#
# def N(x):
#     # x = x.cpu()
#     r = x - x.min()
#     r = r / r.max()
#     return r
#
# def N2(x):
#     # x = x.cpu()
#     # r = x - x.min()
#     r = x / x.max()
#     return r
#
# def Mask(kernel, shift = 0):
#     kernel = kernel.cuda().unsqueeze(-1).expand(size = (2,2,3))
#     mask = tr.zeros_like(A)
#     mask[shift:shift + kernel.shape[0],shift:shift + kernel.shape[1],:] = kernel
#     return mask

def l(x): return tr.log(x)
def e(x): return tr.exp(x)

def les(x): return l((e(x) + 1) / (e(x) - 1))
def sig(x): return tr.sigmoid(x)

def F(A, dim =  (-3,-2)): return Fn(A, dim = dim)
def iF(A, dim = (-3,-2)): return iFn(A, dim = dim)

def lF(A, dim =  (-3,-2)): return tr.log(Fn(A, dim = dim))
def iFe(A, dim = (-3,-2)): return iFn(tr.exp(A), dim = dim)
def Fe(A, dim =  (-3,-2)): return Fn(tr.exp(A.conj()), dim = dim)

def conv(A, B): return iFe(lF(A) + lF(B))
def convs(A): return iFn(tr.exp(tr.log(Fn(r, dim=A.shape[1:-1])).sum(dim=0)) , dim= A.shape[1:-1])

def corr(A, B): return iFe(lF(A) - lF(B))
def delta(A): return corr(A, A)
def inv(A): return corr(delta(A), A)

def N(x):
    # x = x.cpu()

    r = x - x.min()
    r = r / r.max()
    return r

def Nn(x): return x / np.absolute(x).max()
def Nr(x): return x / x.real.max()

def Mask(kernel, A, shift = (0,0,0)):
    kernel = kernel.cuda().unsqueeze(-1).expand(size = tuple(kernel.shape) + (3,))
    mask = tr.zeros_like(A)
    mask[shift[0]:shift[0] + kernel.shape[0], shift[1]:shift[1] + kernel.shape[1], shift[2]:shift[2] + kernel.shape[2]] = kernel
    return mask

def Mask2(kernel, A, shift = (0,0,0)):
    kernel = kernel.cuda().unsqueeze(-1).expand(size = (2,2,3))
    mask = tr.zeros_like(A)
    mask[shift[0]:shift[0] + kernel.shape[0], shift[1]:shift[1] + kernel.shape[1], shift[2]:shift[2] + kernel.shape[2]] = kernel
    return mask

def Mask3(kernel, shapes, shift = (0,0,0)):
    # kernel = kernel.cuda().unsqueeze(-1).expand(size = (2,2,3))
    mask = tr.zeros(shapes)
    mask[shift[0]:shift[0] + kernel.shape[0], shift[1]:shift[1] + kernel.shape[1], shift[2]:shift[2] + kernel.shape[2]] = kernel
    return mask

def save_Fholo(A, B): return lF(A) + lF(B)
# def save_Fholo(A, B): return lF(Conv(A, B))

def load_Fholo(A, holo): return iFe(holo - lF(A))
# def load_Fholo(A, holo): return Corr(iFe(holo), A)


def difr(A, a, B, b, dim=(-1,-2)):
    """Return difraction image A and image B 
    with degree of images diffraction a and b"""
    
    cond1 = A.layout is not torch.strided
    cond2 = B.layout is not torch.strided
    
    if cond1:
        A = A.to_dense()
    if cond2:
        B = B.to_dense()
        
    if A.shape != B.shape:
        A, B = fit_2sizes(A, B)
        
    res = iFe(a * lF(A, dim=dim) + b * lF(B, dim=dim), dim=dim)
    
    if cond1 or cond2:
        res = res.to_sparse()
        
    return res




def cut(x, a = 1, b = 0, imag=False):
    """Cut tensor values upper then a to a, and lower then b to b.
    By default cut imag part of complex values"""
    
    cond = x.layout is not torch.strided
    
    if cond:
        inds = x._indices()
        size = x.size()
        x = x._values()
        
        
    if x.dtype is torch.complex64:
        r = x.real
        r[r>a] = a
        r[r<b] = 0
        if imag:
            i = x.imag
            i[i>a] = a
            i[i<b] = b
            res = tr.complex(r, i)
        else:
            res = r
            
    else:
        x[x>a] = a
        x[x<b] = b
        res = x
        
    if cond:
        mask = (res != 0).nonzero().T.squeeze()
        inds = inds[:, mask]
        vals = res[mask]
        res = tr.sparse_coo_tensor(inds, vals, size=size)
        
    return res
        
    
def holo_diff(data: torch.Tensor, 
              cut_params: dict = {'a':10, 'b':0.2, 'imag':False}
             ) -> torch.Tensor:
    """Return (?holographic?) difference of first dimention (time)
    in 3-dims (video) tensor [time, x, y]"""
    
    cond = data.layout is torch.sparse_coo
    if cond:
        d = data.to_dense()
    else:
        d = data
    res = uth.difr(d[1:], 1, d[:-1], -1)
    res = uth.cut(res, cut_params)
    if cond:
        res = res.to_sparse()
    return res  


def fit_2sizes(A:torch.Tensor, 
              B:torch.Tensor
             ) -> torch.Tensor:
    """Return A and B 2D tensors with equal sizes"""
    
    if A.shape != B.shape:
        shapes = tr.cat([tr.tensor(A.shape).unsqueeze(0), tr.tensor(B.shape).unsqueeze(0)])
        shapes = shapes.max(0).values

        A_ = tr.zeros(*shapes)
        A_[:A.shape[0], :A.shape[1]] = A
        A = A_

        B_ = tr.zeros(*shapes)
        B_[:B.shape[0], :B.shape[1]] = B
        B = B_
        
    return A, B


@tr.jit.script
def norm01(x):
    """Normalize matrix to (0, +1)"""
    
    if x.layout is torch.sparse_coo: 
        inds = x._indices()
        vals = x._values()
        vals -= vals.min()
        vals /= vals.max()
        return coo(inds, vals, size=x.size())
    else:
        x -= x.min()
        x /= x.max()
        return x
    
    
def norm11(x:torch.Tensor
             ) -> torch.Tensor:
    """Normalize matrix to (-1, +1)"""
    
    
    x = norm01(x)
    if x.layout is torch.sparse_coo: 
        inds = x._indices()
        vals = x._values()
        vals = vals * 2 - 1
        return coo(inds, vals, size=x.size())
    else:
        return x * 2 - 1
    
    
    
    
    
        