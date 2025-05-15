import torch
import numpy as np
from pykeops.torch import LazyTensor
from torchmin import minimize as minimize_torch

import itertools # nested loops
import math # binom coefficient
from numpy.polynomial import Legendre, Polynomial, Chebyshev # orthogonal polynomials
from .apds import *

from .log_res_utils import *

class min_diagram_system:
    """
    A minimisation diagram system.
    """
    def __init__(self,
                 Y = None, #pixels / voxels
                 I = None, # grain map
                 pixel_params = None,
                 theta = None, # parameters
                 ho = 2,
                 eps = 1e-2,
                 basis = Legendre,
                 heuristic_guess = True,
                 data_rescaling_type = "centered_unit_interval",
                 dt = torch.float64,
                 device = "cuda" if torch.cuda.is_available() else "cpu",
                 # convenience constructor options:
                 N = 10,
                 D = 2,
                 ani_thres = 0.25,
                 pixel_size_prefactor = 2,
                 seed = -1,
                 ):
        """
        TODO: add description of the init.
        """                         
        self.ho = ho
        self.eps = eps
        self.basis = basis
        self.data_rescaling_type = data_rescaling_type
        self.heuristic_guess = heuristic_guess
        
        self.N = N
        self.D = D
        self.ani_thres = ani_thres
        self.pixel_size_prefactor = pixel_size_prefactor
        
        self.dt = dt
        self.device = device
        torch.set_default_dtype(dt)
        torch.set_default_device(device)

        self.set_grain_map(Y,I,pixel_params)
        self.set_theta(theta)
        self.assemble_design_matrix()

    def update_lr_data(self,
                       #Y = None,
                       ho = None,
                       eps = None,
                       basis = None,
                       #data_rescaling_type = None,
                      ):
        #Y = self.Y if Y is None else Y
        counter = 0
        if basis is not None and basis is not self.basis:
            self.theta = convert_theta_between_bases(self.theta, ho = self.ho, D = self.D, basis_end = basis)
            self.basis = basis
            counter += 1
        if ho is not None and ho <= self.ho:
            print("ho not adjusted -- it only makes sense to increase ho!")
        if ho is not None and ho > self.ho:
            self.theta = reorder_variables(self.theta,self.D,self.ho,ho)
            self.ho = ho
            counter += 1
            
        
        # if data_rescaling_type is not None:
        #     self.data_rescaling_type = data_rescaling_type
        #     self.set_grain_map(Y=Y,I=self.I)
        #     #self.assemble_design_matrix()
        if eps is not None and eps is not self.eps:
            self.eps = eps
            counter += 1

        if counter > 0:
            self.assemble_design_matrix()



    def set_grain_map(self,Y = None, I=None, pixel_params = None):
        if Y is None:
            apd = apd_system(N=self.N,D=self.D,pixel_size_prefactor = self.pixel_size_prefactor, ani_thres = self.ani_thres)
            apd.assemble_pixels()
            Y = apd.Y
            I = apd.assemble_apd()
            pixel_params = apd.pixel_params()
        if self.data_rescaling_type == "centered_unit_interval":
            Y2 = Y - Y.mean((0),keepdim=True)
            Y2 /= Y2.max((0), keepdim=True).values
        elif self.data_rescaling_type == "unit_interval":
            Y2 = Y / Y.abs().max((0),keepdim=True).values
        else:
            Y2 = Y
        self.Y = Y2
        self.I = I
        M, D = self.Y.shape
        self.N = (self.I.max()+1).item()
        self.D = D
        self.M = M
        self.pixel_params = pixel_params
        
        
    def set_theta(self,theta = None):
        if theta is None:
            self.K = math.comb(self.D+self.ho, self.ho)
            if self.heuristic_guess:
                if self.ho < 3:
                    phys_guess = physical_heuristic_guess(self.I,self.Y,ho = self.ho)
                    guess = convert_from_phys_to_lr(phys_guess,ho = self.ho)
                    guess = guess-guess[0,:]
                    self.theta = convert_theta_between_bases(guess,ho=self.ho,D=self.D,basis_end = self.basis)
                else:
                    phys_guess = physical_heuristic_guess(self.I,self.Y,ho=2)
                    guess = convert_from_phys_to_lr(phys_guess,ho=2)
                    guess = guess-guess[0,:]
                    guess = reorder_variables(guess,self.D,2,self.ho)
                    self.theta = convert_theta_between_bases(guess,ho = self.ho,D = self.D,basis_end = self.basis)
            else:
                self.theta = torch.zeros((self.N,self.K))
        else:
            #TODO: add assertion check
            self.theta = theta
            
        self.theta_l = LazyTensor(self.theta.view(self.N,1,self.K))

    def assemble_design_matrix(self):
        self.design_matrix = assemble_design_matrix(self.Y, ho = self.ho,
                                                    basis = self.basis,
                                                    eps = self.eps)
        self.K = self.design_matrix.shape[1]
        self.dml = LazyTensor(self.design_matrix.view(1, self.M, self.K))

    def objective_function(self, theta, backend = "auto"):
        self.theta = theta
        self.theta_l = LazyTensor(self.theta.view(self.N,1,self.K))
        first_sum = torch.sum(torch.index_select(self.theta, 0, self.I)*self.design_matrix)
        second_sum = (self.theta_l * self.dml).sum(dim = 2, backend = backend).logsumexp(dim = 0, backend = backend).sum(dim = 0)
        return -(1 / self.M)*(first_sum - second_sum)


    def fit_theta(self,
                  record_time = False,
                  solver = None,
                  verbose = True,
                  backend = "auto",
                  **kwargs):
        defaultKwargs = {}
        if solver is None:
            solver = 'l-bfgs'
            defaultKwargs = {'gtol': 1e-8,
                             'xtol': -1e-10,
                             'disp': 2 if verbose else 0,
                             'max_iter':10}
        theta_t = torch.zeros(self.theta.shape)
        def fun(theta_red):
            theta_t[1:,:] = theta_red
            return self.objective_function(theta_t, backend = backend)
                    
        # Solve the optimisation problem
        kwargs = { **defaultKwargs, **kwargs }

        start=time.time()
        res = minimize_torch(fun,
                             self.theta[1:,:],
                             method = solver,
                             options = kwargs)
        time_taken = time.time()-start
        if verbose:
            print('It took',time_taken, "seconds to fit theta.")


    def assemble_diagram(self):
        self.theta_l = LazyTensor(self.theta.view(self.N,1,self.K))
        return (-self.theta_l*self.dml).sum(dim=2).argmin(dim=0).ravel()

    
    def plot_diagram(self, color_by = None):
        maxes = self.Y.max((0),keepdims=True).values[0]
        mins = self.Y.min((0),keepdims=True).values[0]
        dom_x = [mins[0],maxes[0]]
        dom_y = [mins[1],maxes[1]]
        domain = torch.tensor([dom_x,dom_y])
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(10.5, 10.5, forward=True)
        I_new = gridify_Y_I(self.Y.cpu(),self.I.cpu(), domain.cpu() , self.pixel_params,
                    color_by = color_by)
        ax[0].imshow(I_new,origin = 'lower')
        I = self.assemble_diagram()
        I_new = gridify_Y_I(self.Y.cpu(),I.cpu(), domain.cpu() , self.pixel_params,
                    color_by = color_by)
        ax[1].imshow(I_new,origin = 'lower')
        return fig, ax

    def apd_from_grain_map(self):
        guess = physical_heuristic_guess(self.I,self.Y,ho = 2)
        moments = calculate_moments_from_data(self.I,self.Y,ho = 2)
        
        As_guess1 = torch.stack((guess[:,0],guess[:,1]),dim=1)
        As_guess2 = torch.stack((guess[:,1],guess[:,2]),dim=1)
        As_guess = torch.stack((As_guess1,As_guess2),dim=2)
    
        #dets =  torch.sqrt(torch.linalg.det(As_guess))
        #As_guess1 = torch.stack((guess[:,0]/dets,guess[:,1]/dets),dim=1)
        #As_guess2 = torch.stack((guess[:,1]/dets,guess[:,2]/dets),dim=1)
        #As_guess = torch.stack((As_guess1,As_guess2),dim=2)
        
        XX = guess[:,3:5]
        maxes = self.Y.max((0),keepdims=True).values[0]
        mins = self.Y.min((0),keepdims=True).values[0]
        dom_x = [mins[0],maxes[0]]
        dom_y = [mins[1],maxes[1]]
        domain = torch.tensor([dom_x,dom_y])
        return apd_system(X = XX.contiguous(),
                                domain = domain,
                                pixel_params = self.pixel_params,
                                As = As_guess.contiguous(),
                                target_masses= (moments[:,0]/len(self.Y)).contiguous(),
                                heuristic_W=True,
                                dt = self.dt)







