import torch
from pykeops.torch import LazyTensor
from torchmin import minimize as minimize_torch
from matplotlib import pyplot as plt

from .geometry_utils import *
from .override_optimality_condition import *

class apd_system:
    """
    An anisotropic power diagram system.
    """
    def __init__(self,
                 domain = None, #rectilinear domains only
                 X = None,
                 As = None,
                 W = None,
                 target_masses = None, #have to add up to 1
                 pixel_params = None,
                 dt = torch.float32,
                 device = "cuda" if torch.cuda.is_available() else "cpu",
                 # convenience constructor options:
                 N = 10,
                 D = 2,
                 ani_thres = 0.25,
                 heuristic_W = False,
                 radius_of_exclusion = 0.01,
                 det_constraint = True,
                 error_tolerance = 0.01,
                 pixel_size_prefactor = 2, # consider increasing or having a seperate prefactor for plotting apds
                 seed = -1,
                 ):
        """
        TODO: add description of the init.
        """
        
        self.N = N
        self.D = D

        self.dt = dt
        self.device = device
        torch.set_default_dtype(dt)
        #torch.set_default_device(device)
        self.error_tolerance = error_tolerance
        self.radius_of_exclusion = radius_of_exclusion
        self.ani_thres = ani_thres
        self.seed = seed
        self.det_constraint = det_constraint
        self.heuristic_W = heuristic_W
        self.pixel_size_prefactor = pixel_size_prefactor
        
        self.set_domain(domain)

        self.set_X(X)
        
        self.set_As(As)
        self.set_target_masses(target_masses)
        
        self.set_W(W)
        
        self.set_pixel_params(pixel_params)

                
        self.Y = None
        self.y = None
        self.PS = None
        self.optimality = False
        self.data = {}
        
    
        
    def set_domain(self,domain = None):
        """
        TODO: add description.
        """
        if domain is None:
            domain = torch.tensor([[0,1]]*self.D)
        else:
            self.D = domain.shape[0]
            
        self.domain = domain.to(device=self.device, dtype=self.dt)
        self.box_size = torch.prod(self.domain[:,1] - self.domain[:,0]).to(device = self.device,dtype=self.dt)
        
        
        
    def set_X(self, X = None, verbose = False):        
        """
        TODO: add description.
        """
        if X is None:
            if not ( self.seed == -1):
                torch.manual_seed(self.seed)
            
            X = sample_seeds_with_exclusion(self.N,dim=self.D,radius_prefactor = self.radius_of_exclusion,verbose = verbose).to(device=self.device, dtype=self.dt)
            X = self.domain[:,0] + (self.domain[:,1]-self.domain[:,0]) * X
        else:
            self.N = len(X)
            self.D = X.shape[-1]

        self.X = X.to(device=self.device, dtype=self.dt)
        self.x = LazyTensor(self.X.view(self.N, 1, self.D))  # (N, 1, D)
        

    def set_As(self, As = None):
        """
        TODO: add description.
        """
        if As is None:
            if not ( self.seed == -1):
                torch.manual_seed(100+self.seed)
            if self.det_constraint:
                As = sample_normalised_spd_matrices(self.N,dim=self.D,ani_thres = self.ani_thres)
            else:
                As = sample_psd_matrices_perturbed_from_identity(self.N,dim=self.D,amp = self.ani_thres)
        
        self.As = As.to(device=self.device, dtype=self.dt)
        self.a = LazyTensor(self.As.view(self.N, 1, self.D * self.D))
        
        
    def set_target_masses(self,target_masses = None):
        """
        TODO: add description.
        """
        if target_masses is None:
            target_masses = specify_volumes(self.N)
            
        self.target_masses = self.box_size*target_masses.to(device=self.device, dtype=self.dt)

        
    def set_W(self,W = None):
        """
        TODO: add description.
        """
        if W is None:
            if self.heuristic_W:
                W = initial_guess_heuristic(self.As,self.target_masses,self.D)
            else:
                W = torch.zeros(self.N)
        
        self.W = W.to(device=self.device, dtype=self.dt)
        self.w = LazyTensor(self.W.view(self.N,1,1))
        
    def set_pixel_params(self,pixel_params = None,verbose=False):
        """
        TODO: add description.
        """
        if pixel_params is None:
            M = (self.box_size/(self.error_tolerance*torch.min(self.target_masses)))**(1/self.D)
            M = int(self.pixel_size_prefactor*int(M.item()))
            pixel_params = (M,)*self.D    
        self.pixel_params = pixel_params
        if verbose:
            print("M = ",pixel_params)

              

        

    
    def assemble_pixels(self):
        """
        Assemble the pixels/voxels explicitly. 
        """
        i = 0
        grid_x = torch.linspace(self.domain[i,0] + 0.5 * (self.domain[i,1] - self.domain[i,0]) / self.pixel_params[i], self.domain[i,1] - 0.5 * (self.domain[i,1] - self.domain[i,0])/self.pixel_params[i], self.pixel_params[i])
        i = 1
        grid_y = torch.linspace(self.domain[i,0] + 0.5 * (self.domain[i,1] - self.domain[i,0])/self.pixel_params[i], self.domain[i,1] - 0.5 * (self.domain[i,1] - self.domain[i,0])/self.pixel_params[i], self.pixel_params[i])
        if self.D == 3:
            i = 2
            grid_z = torch.linspace(self.domain[i,0] + 0.5 * (self.domain[i,1] - self.domain[i,0])/self.pixel_params[i], self.domain[i,1] - 0.5 * (self.domain[i,1] - self.domain[i,0])/self.pixel_params[i], self.pixel_params[i])

        mesh = torch.meshgrid((grid_x,grid_y), indexing="ij") if self.D == 2 else torch.meshgrid((grid_x,grid_y,grid_z), indexing="ij")
        
        pixels = torch.stack(mesh, dim=-1).to(device=self.device,dtype=self.dt)
        pixels = pixels.reshape(-1, self.D)
        
        pixel_vols = (self.domain[:,1] - self.domain[:,0])/torch.tensor(self.pixel_params).to(device=self.device,dtype=self.dt)
        
        i = 0
        PS_x = pixel_vols[i]*torch.ones(self.pixel_params[i]).to(device=self.device,dtype=self.dt)
        i = 1
        PS_y = pixel_vols[i]*torch.ones(self.pixel_params[i]).to(device=self.device,dtype=self.dt)
        
        if self.D == 3:
            i = 2
            PS_z = pixel_vols[i]*torch.ones(self.pixel_params[i]).to(device=self.device,dtype=self.dt)
        

        PS = (PS_x[:, None] @ PS_y[None,:])
        if self.D == 3:
            PS = PS[:,:,None] @ PS_z[None,:]

        PS = PS.reshape(-1,1).flatten()
        self.Y = pixels
        self.PS = PS
        self.y = LazyTensor(self.Y.view(1, torch.prod( torch.tensor( self.pixel_params ) ).item(), self.D)) 
        
                    
    def assemble_apd(self,record_time = False,verbose=False,
                    color_by = None, backend = "auto"):
        """
        Assemble the apd by finding which grain each pixels belongs to.
        """
        if self.Y is None:
            self.assemble_pixels()
        start=time.time()
        D_ij = ((self.y - self.x) | self.a.matvecmult(self.y - self.x)) - self.w
        # Find which grain each pixel belongs to
        grain_indices = D_ij.argmin(dim=0,backend=backend).ravel() # grain_indices[i] is the grain index of the i-th voxel
        time_taken = time.time()-start
        if record_time:
            self.apd_gen_time = time_taken
        
        if verbose:
            print("APD generated in:", time_taken, "seconds.")
        if color_by is not None:
            return color_by[grain_indices]#.reshape(self.pixel_params)
        else:
            return grain_indices#.reshape(self.pixel_params)
    
    def plot_apd(self, color_by = None):
        """
        Plot the APD (for now only in 2D). 
        """
        if self.D == 2:
            img = self.assemble_apd(color_by=color_by).reshape(self.pixel_params).transpose(0,1).cpu()
            fig, (ax1) = plt.subplots(1,1)
            fig.set_size_inches(10.5, 10.5, forward=True)
            ax1.imshow(img, origin='lower', extent = torch.flatten( self.domain ).tolist())
            return fig, ax1
        
    def plot_ellipses(self):
        if self.D == 2:
            decomp = torch.linalg.eigh(self.As)
            AB = decomp.eigenvalues**(-0.5)
            scaling = (self.target_masses/torch.pi)**(1/2)
            AB[:,0] = AB[:,0] * scaling
            AB[:,1] = AB[:,1] * scaling
            Rots = decomp.eigenvectors
            t = torch.linspace(0, 2*torch.pi, 80).to(device=self.device,dtype = self.dt)
            Ell1 = AB[:,0,None] @ torch.cos(t)[None,:]
            Ell2 = AB[:,1,None] @ torch.sin(t)[None,:]

            Ell = torch.stack([Ell1,Ell2])

            Ell = (Ell.transpose(0,2)).transpose(0,1)
            Ell_rot = Ell @ Rots
            Ell_rot_shifted = [Ell_rot[i] + self.X[i] for i in range( len(Ell_rot))]
            fig, (ax1) = plt.subplots(1,1)
            fig.set_size_inches(10.5, 10.5, forward=True)
            for i in range(0,len(Ell_rot_shifted)):
                ax1.scatter(self.X.cpu()[i,0],self.X.cpu()[i,1],c='r',s=3)
                ax1.plot(Ell_rot_shifted[i].cpu()[:,0],Ell_rot_shifted[i].cpu()[:,1],c='k')
            ax1.set_xlim(self.domain[0].cpu().numpy()) 
            ax1.set_ylim(self.domain[1].cpu().numpy()) 
            return fig, ax1
            
        
    
    def OT_dual_function(self, W, backend = "auto"):
        """
        Helper function for assembling the OT dual function g(W).
        """
        #if not ( W is None ):
        self.W = W
        self.w = LazyTensor(self.W.view(self.N,1,1))
        
        #if self.Y is None:
        #    self.assemble_pixels()
        
        D_ij = ((self.y - self.x) | self.a.matvecmult(self.y - self.x)) - self.w
        idx = D_ij.argmin(dim=0,backend=backend).view(-1)
        #A[idx] is designed to be determistic, so slow
        
        # profile the performance for the indexing A[idx] vs torch.index_select(A,0,idx)
        ind_select = torch.index_select(self.X,0,idx)-self.Y
        MV = torch.einsum('bij,bj->bi', torch.index_select(self.As,0,idx), ind_select)
        sD_ij = torch.einsum('bj,bj->b',MV,ind_select) - torch.index_select(self.W, 0, idx)
        g=-torch.dot(self.target_masses,self.W)-torch.dot(sD_ij,self.PS)
        return g
    
    
    def check_optimality(self,
                         error_wrt_each_grain = True,
                         return_gradient_and_error = False,
                         backend = "auto"):
        """
        Check whether the APD generated by (X,As,W) is optimal with respect to target masses.
        """
        if self.Y is None:
            self.assemble_pixels()
        
        D_ij = ((self.y - self.x) | self.a.matvecmult(self.y - self.x)) - self.w
        
        grain_indices = D_ij.argmin(dim=0, backend = backend).ravel()
        volumes = torch.bincount(grain_indices,self.PS,minlength=self.N)
        Dg = volumes - self.target_masses
        if error_wrt_each_grain:
            error = torch.max(torch.abs(Dg)/self.target_masses)
        else:
            error = torch.max(torch.abs(Dg)/torch.min(self.target_masses))
                              
        if error < self.error_tolerance + 1e-5:
            print("The APD is optimal!")
            print('Percentage error = ',(100*error).item())
            self.optimality = True
        else:
            print("Precision loss detected!")
            print('Percentage error = ',(100*error).item())
        if return_gradient_and_error:
            return Dg, error
        
    
    def find_optimal_W(self,
                       record_time = False,
                       error_wrt_each_grain = True,
                       solver = None,
                       verbose = True,
                       backend = "auto",
                             **kwargs):
        """
        Find the set of weights W for which the APD generated by (X,As,W) is optimal with respect to target masses. 
        """
        if error_wrt_each_grain:
            vol_tol = (self.error_tolerance)*self.target_masses
            if verbose:
                print("Solver tolerance is with respect to each grain separately.")
                print("Smallest tol: ", torch.min(vol_tol))
        else:
            vol_tol = self.error_tolerance*torch.min(self.target_masses)
            if verbose:
                print("Solver tolerance is with respect to the smallest grain.")
                print("Tolerance: ", vol_tol)

        
        defaultKwargs = {}
        if solver is None:
            solver = 'bfgs'
            defaultKwargs = {'gtol': vol_tol, 'xtol': 0, 'disp': 1 if verbose else 0, 'max_iter':1000}

            
        fun = lambda W : self.OT_dual_function(W=W,backend=backend)
        
        # Solve the optimisation problem
        kwargs = { **defaultKwargs, **kwargs }
        
        start=time.time()
        res = minimize_torch(fun, self.W, method=solver, disp= 1 if verbose else 0,
                             options=kwargs)
        time_taken = time.time()-start
        if verbose:
            print('It took',time_taken, "seconds to find optimal W.")
        
        W=res.x
        
        self.W = W
        self.w = LazyTensor(self.W.view(self.N,1,1))
        if record_time:
            self.time_to_find_W = time_taken
        
        
    def adjust_X_As(self,backend="auto",adjust_As = False):
        if not self.optimality:
            print("Find optimal W first!")
        else:
            D_ij = ((self.y - self.x) | self.a.matvecmult(self.y - self.x)) - self.w
            grain_indices = D_ij.argmin(dim=0,backend=backend).ravel()
            new_X0 = torch.bincount(grain_indices, self.Y[:,0], minlength = self.N)
            new_X1 = torch.bincount(grain_indices, self.Y[:,1], minlength = self.N)

            normalisation = torch.bincount(grain_indices,minlength=self.N)
            
            if self.D == 3:
                new_X2 = torch.bincount(grain_indices,self.Y[:,2],minlength=self.N)
                self.X = torch.stack([new_X0/normalisation, new_X1/normalisation,new_X2/normalisation],dim=1)
            else:
                self.X = torch.stack([new_X0/normalisation, new_X1/normalisation],dim=1)

            if adjust_As:
                YY_XX_new = self.Y - self.X[grain_indices]
                tensor_prod = torch.einsum('bc,bd->bcd', YY_XX_new, YY_XX_new)
                a00 = torch.bincount(grain_indices, tensor_prod[:,0,0], minlength=self.N)
                a01 = torch.bincount(grain_indices, tensor_prod[:,0,1], minlength=self.N)
                a11 = torch.bincount(grain_indices, tensor_prod[:,1,1], minlength=self.N)
                As0_new = torch.stack([4.0*a00/normalisation, 4.0*a01/normalisation],dim=1)
                As1_new = torch.stack([4.0*a01/normalisation, 4.0*a11/normalisation],dim=1)
                As_new = torch.linalg.inv(torch.stack([As0_new, As1_new],dim=1))
                ee,vv = torch.linalg.eigh(As_new)
                ratios = ee[:,1] / ee[:,0]
                major_axes = torch.sqrt(ratios)
                minor_axes = 1/major_axes
                ee_normalised = torch.stack([major_axes,minor_axes],dim=1)
                self.As = (vv @ torch.diag_embed(ee_normalised) @ vv.mT.conj())
                self.a = LazyTensor(self.As.view(self.N, 1, self.D * self.D))

            self.optimality = False
            self.x = LazyTensor(self.X.view(self.N, 1, self.D))
            
            
    def Lloyds_algorithm(self,K=5, verbosity_level = 1, backend = "auto",adjust_As = False):
        for k in range(K):
            if verbosity_level > 0:
                print("Lloyds iteration:", k)
            
            verbose = True if verbosity_level == 2 else False
            self.find_optimal_W(verbose = verbose)
            self.check_optimality()
            self.adjust_X_As(backend = backend, adjust_As = adjust_As)
        