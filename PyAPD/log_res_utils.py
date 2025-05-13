import torch
import numpy as np
from scipy.interpolate import griddata
from numpy.polynomial import Legendre, Polynomial, Chebyshev # orthogonal polynomials
import itertools # nested loops
import math # binom coefficient

def assemble_design_matrix(Y,
                           ho=2,
                           basis = Legendre,
                           eps = 1.0):
    M, D = Y.shape
    Y_d = None
    for d in range(D):
        Y_d_t = Y[:,d][:,None].expand(-1,ho+1).detach().clone()
        for a in range(0,ho+1):
            Y_d_t[:,a] = torch.from_numpy(basis.basis(a)(Y_d_t[:,a]))
        if Y_d is None:
            Y_d = Y_d_t
        else:
            Y_d = torch.stack([Y_d,Y_d_t],axis=2)
    K = math.comb(D+ho, ho)
    design_matrix = torch.zeros(M,K)
    ii = 0
    for indices in itertools.product(range(ho+1), repeat=D):
        if sum(indices) < ho+1:
            #design_matrix[:,ii] = Y_d[:,indices[0],0]*Y_d[:,indices[1],1]
            design_matrix[:,ii] = math.prod([Y_d[:,indices[d],d] for d in range(D)])
            ii += 1

    return design_matrix / eps


def reorder_variables(theta, # N x K Torch tensor where N is the number of grains, K is the number of features
                      D, # dimension of the problem (technically can be recovered from K and ho_start, but kept for convenience) 
                      ho_start, # highest order of polynomial features in theta, ho_start and D determines K 
                      ho_end, # target highest order of polynomial features
                     ):
    N, K_start = theta.shape
    K_end = math.comb(D+ho_end, ho_end)
    #theta_new = torch.cat((results[0],torch.zeros(N,K_end-K_start)),dim=1)
    theta_new = torch.zeros(N,K_end)
    I1 = [indices for indices in itertools.product(range(ho_end+1),repeat=D) if sum(indices) < ho_end+1]
    I2 = [indices for indices in itertools.product(range(ho_start+1),repeat=D) if sum(indices) < ho_start+1]
    II = [ I1.index(i) for i in I2 ]
    theta_new[:,II] = theta
    return theta_new


def calculate_moments_from_data(I,
                                Y,
                                ho = 2,
                                basis = Polynomial,
                               ):
    N = (I.max()+1).item()
    M, D = Y.shape
    DM = assemble_design_matrix(Y,ho=ho, basis = basis)
    _ , K = DM.shape
    moments = torch.zeros(N,K)
    for k in range(K):
        moments[:,k] = torch.bincount(I,DM[:,k],minlength=N)
    return moments


# This is between physical variables and a monomial basis (`Polynomial` basis)
def convert_from_lr_to_phys(theta,ho=1):
    assert ho > 0
    phys_var = torch.zeros(theta.shape)
    if ho == 1:
        phys_var[:,0] = theta[:,2]/2.0
        phys_var[:,1] = theta[:,1]/2.0
        phys_var[:,2] = theta[:,0] + 0.25*(theta[:,2]**2 + theta[:,1]**2)
    if ho == 2:
        phys_var[:,0] = -theta[:,-1]
        phys_var[:,1] = -theta[:,-2]/2.0
        phys_var[:,2] = -theta[:,2]
        A1 = torch.stack((phys_var[:,0],phys_var[:,1]),dim=1)
        A2 = torch.stack((phys_var[:,1],phys_var[:,2]),dim=1)
        A = torch.stack((A1,A2),dim=2)
        phys_var[:,(3,4)] = torch.linalg.solve(A,theta[:,(3,1)])/2.0
        phys_var[:,5] = theta[:,0] + (phys_var[:,3]*theta[:,3] 
                                      + phys_var[:,4]*theta[:,1])/2.0
    if ho > 2:
        print("No conversation implemented, returing zeros")
    return phys_var
        

def convert_from_phys_to_lr(phys_var,ho=1):
    assert ho > 0
    theta = torch.zeros(phys_var.shape)
    if ho == 1:
        theta[:,0] = phys_var[:,2] - torch.norm(phys_var[:,0:2],dim=1)**2
        theta[:,1] = 2.0*phys_var[:,1]
        theta[:,2] = 2.0*phys_var[:,0]
        return theta
    if ho == 2:
        theta[:,0] = phys_var[:,-1] - (phys_var[:,0]*phys_var[:,3]**2
                                       + 2.0*phys_var[:,1]*phys_var[:,3]*phys_var[:,4]
                                       + phys_var[:,2]*phys_var[:,4]**2)
        theta[:,1] = 2.0*(phys_var[:,1]*phys_var[:,3] + phys_var[:,2]*phys_var[:,4])
        theta[:,2] = -phys_var[:,2]
        theta[:,3] = 2.0*(phys_var[:,0]*phys_var[:,3] + phys_var[:,1]*phys_var[:,4])
        theta[:,4] = -2.0*phys_var[:,1]
        theta[:,5] = -phys_var[:,0]
        return theta
    if ho > 2:
        print("No conversation implemented, returing zeros")
        return theta
        

def physical_heuristic_guess(I,
                             Y,
                             ho = 2
                            ):
    _, D = Y.shape
    moments = calculate_moments_from_data(I,
                                Y,
                                ho = ho)
    if ho > 2:
        print("No guess implemented, returning zeros")
        return torch.zeros(moments.shape)
    elif ho == 2:                    
        y1 = moments[:,3]/moments[:,0]
        y2 = moments[:,1]/moments[:,0]
        b11 = (moments[:,-1] - y1*moments[:,3])/moments[:,0]
        b12 = (moments[:,-2] - y1*moments[:,1])/moments[:,0]
        b22 = (moments[:,2] - y2*moments[:,1])/moments[:,0]
        B1 = torch.stack((b11,b12),dim=1)
        B2 = torch.stack((b12,b22),dim=1)
        B = 16.0*torch.stack((B1,B2),dim=2)
        A = torch.linalg.inv(B)
        vols = moments[:,0] / Y.shape[0]
        dets = torch.linalg.det(A)
        prefactor = (1.0/torch.pi) if D == 2 else 3.0/(4.0*torch.pi) 
        w = (prefactor * vols * torch.sqrt(dets))**(2.0/D)
        return torch.stack((A[:,0,0],A[:,0,1],A[:,1,1],y1,y2,w),dim=1)
    elif ho == 1:
        y1 = moments[:,2]/moments[:,0]
        y2 = moments[:,1]/moments[:,0]
        w = moments[:,0] / (Y.shape[0] * torch.pi)
        return torch.stack((y1,y2,w),dim=1)


def convert_theta_between_bases(theta,ho,D,basis_start = Polynomial, basis_end = Legendre):
    j = 0
    aa_all = torch.zeros(theta.shape[1],theta.shape[1])
    for i in itertools.product(range(ho+1), repeat=D):
        if sum(i) < ho+1:
            aa = np.zeros(len(theta[0]))
            a1 = np.zeros(ho+1)
            a2 = np.zeros(ho+1)
            
            alpha1 = basis_end.cast(basis_start.basis(i[0])).coef
            alpha2 = basis_end.cast(basis_start.basis(i[1])).coef
            a1[:alpha1.shape[0]] = alpha1
            a2[:alpha2.shape[0]] = alpha2
            
            kk = 0
            for k in itertools.product(range(ho+1),repeat=D):
                if sum(k) < ho+1:
                    aa[kk] = a1[k[0]]*a2[k[1]]
                    kk += 1
            aa_all[:,j] = torch.tensor(aa)
            j += 1
    return  torch.einsum('bi,ji->bj', theta, aa_all)


def gridify_Y_I(Y,I,
                domain,
                pixel_params,
                color_by = None):
    x = Y[:,0]
    y = Y[:,1]
    z = color_by[I] if color_by is not None else I
    # define grid
    xi = np.linspace(domain[0][0],domain[0][1],pixel_params[0])
    yi = np.linspace(domain[1][0],domain[1][1],pixel_params[1])
    # grid the data
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='nearest')
    return zi












    
        