import torch
import numpy as np
from pykeops.torch import LazyTensor

# def sample_grid(n_x, dim=3,x_length = 1, n_y = None, y_length = None, n_z = None, z_length = None):
#     """ Sample n_x * n_y * n_z points along a regular grid on [0,x_length] * [0,y_length] * [0,z_length].
#     Used as an empirical measure approximation of the input measure.
    
#     Input arguments:
#     n_x - number of sampled point along x direction (necessary)    
    
#     dim - dimension (optional, default is 3).

#     x_length - length of the box in x direction (optional, default is x_length = 1)
    
#     n_y - number of sampled points along y direction (optional, default is n_y = n_x)
    
#     y_length - length of the box in y direction (optional, default is y_length = x_length)
     
#     n_z - number of sampled points along z direction (optional, default is n_z = None if dim = 2 and n_z = n_x if dim = 3)
    
#     z_length = length of the box in z direction (optional, default is z_length = None if dim = 2 and z_length = x_length if dim =2)
#     """
#     if n_y == None:
#         n_y = n_x
    
#     if y_length == None:
#         y_length = x_length

#     if n_z == None and dim == 3:
#         n_z = n_x
    
#     if z_length == None and dim == 3:
#         z_length = x_length
    
    
#     grid_points_x = torch.linspace(0.5*x_length/n_x, x_length-0.5*x_length/n_x, n_x)
#     grid_points_y = torch.linspace(0.5*y_length/n_y, y_length-0.5*y_length/n_y, n_y)
#     if dim == 3:
#         grid_points_z = torch.linspace(0.5*z_length/n_z, z_length-0.5*z_length/n_z, n_z)
        
#     mesh = torch.meshgrid((grid_points_x,grid_points_y), indexing="ij") if dim == 2 else torch.meshgrid((grid_points_x,grid_points_y,grid_points_z), indexing="ij")
    
#     grid_points = torch.stack(mesh, dim=-1)
#     grid_points = grid_points.reshape(-1, dim)
#     volumes_x = (x_length/n_x)
#     volumes_y = (y_length/n_y)
#     if dim == 3:
#         volumes_z = (z_length/n_z)
    
#     PS_x = volumes_x*torch.ones(n_x)
#     PS_y = volumes_y*torch.ones(n_y)
#     if dim == 3:
#         PS_z = volumes_z*torch.ones(n_z)

#     PS = (PS_x[:, None] @ PS_y[None,:])
#     if dim == 3:
#         PS = PS[:,:,None] @ PS_z[None,:]
    
#     PS = PS.reshape(-1,1).flatten()
        
#     return grid_points, PS



# def sample_uniform(n, dim=3):
#     """ Sample n points from a uniform distribution on [0,1]^dim.
#     Used as seed points for the grains. 
    
#     Input arguments:
    
#     n - number of seed points
    
#     dim - dimension (optional, default is dim = 3).
#     """
#     return torch.rand(n, dim)

def sample_seeds_with_exclusion(n, dim=3,radius_prefactor = 1e-1,number_of_attempts = None, verbose = False):
    """ Sample n points from a uniform distribution on [0,1]^dim, but disregard points too close to each other.
    Used as seed points for the grains. 
    Input arguments:
    
    n - number of seed points
    dim - dimension (optional, default is dim = 3)
    radius_prefactor - prefactor in front of the radius of exclusion which is given by radius_prefactor * n**(-1/dim) (optional, default is radius_prefactor =  0.1)
    number_of_attemps - how many times to try to generate a new seed point 
    verbose - whether to print out how many proposed seed points have been rejected (optional, default is verbose = False)
    """
    if number_of_attempts is None:
        number_of_attempts = 100*n
    
    radius = radius_prefactor * (n**(-1/dim)) 
    X = torch.rand(1,dim)
    counter = 0
    while len(X) < n:
        x_new = torch.rand(1,dim)
        if torch.min(torch.norm(X - x_new,dim=1)) > radius:
            X = torch.cat((X,x_new),dim=0)
        else:
            counter += 1
            if counter > number_of_attempts:
                print("Radius too large to have feasible chance of generating a sample")
                print("Only", len(X), "seed points have been generated.")
                break
    
    if verbose:
        print(counter, "proposed seed points have been excluded.")
        
    return X

# def sample_psd_matrices(n, dim=3):
#     """Generate a collection of n unnormalised n random dim x dim positive semi-definite matrices.
#     Used to specify preferred orientation and aspect ratio of each grain.
    
#     Input arguments:
    
#     n - number of matrices
    
#     dim - dimension (optional, default is dim = 3).
#     """
#     a = torch.randn(n, dim, dim)
#     #a = torch.diag_embed(torch.ones(n, dim))
#     a = 0.5 * (a @ a.transpose(-1, -2))
#     assert a.shape == (n, dim, dim)
#     return a

# def generate_identity_matrices(n,dim=3):
#     """Generate a collection of (dim x dim) identity matrices.
#     Used to specify the isotropy of each grain.
    
#     Input arguments:
    
#     n - number of matrices
    
#     dim - dimension (optional, default is dim = 3).
#     """
#     a=torch.stack((torch.eye(dim),)* n)
#     a.reshape(n,dim,dim)
#     return a

def sample_spd_matrices_perturbed_from_identity(n,dim=3,amp=0.01):
    """Generate a collection of dim x dim spd matrices.
    Used to specify the anisotropy of each grain.
    
    Input arguments:
    
    n - number of matrices
    
    dim - dimension (optional, default is dim = 3).
    
    amp - amplitute of the perturbation from identity (optional, default is amp=0.01). 
    """
    a=torch.stack((torch.eye(dim),)*n)
    a.reshape(n,dim,dim)
    a=a+amp*(2*torch.rand(n,dim,dim)-1)
    a = (a @ a.transpose(-1, -2))
    return a


def convert_axes_and_angle_to_matrix_2D(a,b,theta):
    """Given the semi-major and semi-minor axis of an ellipse and an angle orientation, return the  2x2 positive semi-definite matrix associated with it.
    It can be used to generate 2D normalised anistropy matrices from 2D EBSD data.
    
    Input arguments:
    
    a - major axis (necessary)
    
    b - minor axis (necessary)
    
    theta - orientation angle (necessary)
    """
    a11 = (1/a**2)*np.cos(theta)**2 + (1/b**2)*np.sin(theta)**2
    a22 = (1/a**2)*np.sin(theta)**2 + (1/b**2)*np.cos(theta)**2
    a12 = ((1/a**2) - (1/b**2))*np.cos(theta)*np.sin(theta)
    A = torch.tensor([[a11, a12], [a12, a22]])
    return A

def sample_normalised_spd_matrices(N, dim = 3, ani_thres=0.5):
    """ Generate a collection of n normalised random dim x dim symmetric positive definite matrices.
    Used to specify preferred orientation and aspect ratio of each grain, while retaining the normalisation constraint (determinant equal to 1). 
    
    Input arguments:
    n - number of matrices
    dim - dimension (optional, default is dim = 3).
    ani_thres - acceptable level of anisotropy, values between [0,1], close to 1 means we accept any level of anisotropy, close to 0 that we want next no anisotropy (optional, default is 0.5). Note that setting this parameter 1 can make the resulting optimisation problem ill-conditioned.
    """
    if dim == 2:
        a_s = torch.ones(N) if ani_thres == 0 else torch.distributions.Uniform(1-ani_thres,1).sample([N])
        b_s = 1.0/a_s
        thetas = torch.distributions.Uniform(0,2*np.pi).sample([N])

        ss = torch.sin(thetas)
        cc = torch.cos(thetas)
        rots = torch.stack([torch.stack([cc, -ss],dim=1),
                            torch.stack([ss, cc], dim=1)],dim=2)
        IIs = torch.stack([torch.stack([1/a_s**2,torch.tensor([0.0]*N)],dim=1),
                          torch.stack([torch.tensor([0.0]*N),1/b_s**2],dim=1)],dim=2)
        As = rots @ IIs @ torch.transpose(rots,1,2)
    else:
        a_s = torch.ones(N) if ani_thres == 0 else torch.distributions.Uniform(1-ani_thres,1).sample([N])
        b_s = torch.ones(N) if ani_thres == 0 else torch.distributions.Uniform(1-ani_thres,1.0/(1-ani_thres)).sample([N])
        c_s = 1.0/(a_s*b_s)
        alphas = torch.distributions.Uniform(0,2*np.pi).sample([N])
        betas = torch.distributions.Uniform(0,2*np.pi).sample([N])
        gammas = torch.distributions.Uniform(0,2*np.pi).sample([N])
        
        ss = torch.sin(alphas)
        cc = torch.cos(alphas)

        rots_x = torch.stack([torch.stack([torch.ones(N),
                                           torch.zeros(N),
                                           torch.zeros(N)],dim=1),
                              torch.stack([torch.zeros(N), cc, -ss], dim=1),
                              torch.stack([torch.zeros(N), ss, cc],dim=1)],dim=2)
        ss = torch.sin(betas)
        cc = torch.cos(betas)

        rots_y = torch.stack([torch.stack([cc, torch.zeros(N), ss],dim=1),
                              torch.stack([torch.zeros(N),
                                           torch.ones(N),
                                           torch.zeros(N)], dim=1),
                              torch.stack([-ss ,torch.zeros(N), cc],dim=1)],dim=2)
        
        ss = torch.sin(gammas)
        cc = torch.cos(gammas)

        rots_z = torch.stack([torch.stack([cc, -ss, torch.zeros(N)],dim=1),
                              torch.stack([ss, cc, torch.zeros(N)], dim=1),
                              torch.stack([torch.zeros(N),
                                           torch.zeros(N), 
                                           torch.ones(N)],dim=1)],dim=2)
        
        rots = rots_z @ rots_y @ rots_x
        
        IIs = torch.stack([torch.stack([1/a_s**2, 
                                        torch.zeros(N), 
                                        torch.zeros(N)],dim=1),
                           torch.stack([torch.zeros(N),
                                        1/b_s**2,
                                        torch.zeros(N)],dim=1),
                           torch.stack([torch.zeros(N),
                                        torch.zeros(N),
                                        1/c_s**2],dim=1)],dim=2)

        As = rots @ IIs @ torch.transpose(rots,1,2)
    return As


# def sample_normalised_spd_matrices_2D(n,anisotropy_threshold=0.5):
#     """ Generate a collection of n normalised random 2x2 symmetric positive definte matrices.
#     Used to specify preferred orientation and aspect ratio of each grain, while retaining the normalisation constraint (determinant equal to 1), which is achieved by setting the minor axis to be a multiplicative inverse of the major axis. 
    
#     Input arguments:
#     n - number of matrices
#     anisothropy threshold - acceptable level of anisotropy, values between [0,1], close to 1 means we accept any level of anisotropy, close to 0 that we want next no anisotropy (optional, default is 0.5). Note that setting this parameter 1 can make the resulting optimisation problem ill-conditioned.
#     """
#     #assert anisotropy_threshold < 1
#     #if anisotropy_threshold == 0:
#     #        aa = np.array([1]*n)
#     #else
        
#     #assert anisotropy_threshold > 0
#     As = torch.empty((n,2,2))
#     aa = np.array([1]*n) if anisotropy_threshold == 0 else torch.distributions.uniform.Uniform(1-anisotropy_threshold,1).sample([n]).cpu() 
#     tt = torch.distributions.uniform.Uniform(0,2*np.pi).sample([n]).cpu() 
#     for i in range(n):
#         As[i] = convert_axes_and_angle_to_matrix_2D(aa[i],1/aa[i],tt[i])
#         # check if the determinant is approximately 1:
#         #assert abs(np.linalg.det(As[i].cpu()) - 1.0) < 1e-2
#     return As

def initial_guess_heuristic(As,TVs,D):
    """ Compute the initial guess for the weights based on the heuristic from 
     Kirubel Teferra & David J. Rowenhorst (2018)
     Direct parameter estimation for generalised balanced power diagrams,
     Philosophical Magazine Letters, 98:2, 79-87, 
     DOI: 10.1080/09500839.2018.1472399 
    
    Input arguments:
    As - set of anisotropy matrices (necessary)
    TVs - set of target volumes of grains (necessary)
    D - dimension of the problem (necessary)
    """
    dets = torch.linalg.det(As)
    prefactor = (1.0/torch.pi) if D == 2 else 3.0/(4.0*torch.pi) 
    return (prefactor * TVs * torch.sqrt(dets))**(2.0/D)



def specify_volumes(n,
    crystal_types = 1,
    ratios=None,
    volume_ratios = None,
    max_volume_deviation = 10):
    """ Specify the target volumes of the grains. Grains can be grouped into T types (default T = 1)
    
    Input arguments:
    
    n - number of grains (necessary)
    
    crystal_types - number of grain types (optional, default is crystal_types = 1) 
    ratios - approximate percentage number of grains of each type, e.g. ratios = [0.2, 0.5, 0.3] for 3 graintypes
    (optional, default is a random vector sampled from uniform distribution over the open standard (crystal_types − 1)-simplex.
    
    volume_ratios - e.g. volume_ratios = [1,4,5] for 3 grain types to indicate that grains of type 2 should be 4-times the size of grains of type 1
    and grains of type 3 should be 5 times the size of grains of type 1
    (optional, default is a random vector sampled from a Categorical distribution between 1 and max_volume_deviation
    
    max_volume_deviation - explained above (optional, default max_volume_deviation = 10)
    """
    if ratios == None:
        probs=torch.ones(crystal_types)
        c=torch.distributions.Dirichlet(probs)
        ratios = c.sample()
    
    rand = torch.rand(n)
    crystal = torch.ones(n)
    for k in range(crystal_types):
        crystal[sum(ratios[:k]) <= rand] = k+1
        
    if volume_ratios == None:
        probs=torch.tensor([[1]*max_volume_deviation]*crystal_types)
        c=torch.distributions.Categorical(probs)
        volume_ratios = ratios = c.sample() + 1        

    volumes = torch.ones(n)
    for k in range(crystal_types):
        volumes[crystal == k+1] = volume_ratios[k]

    return volumes / volumes.sum()



# def find_centroids(*,D,N,M,X,
#              x_length = 1,
#              y_length = None,
#              z_length = None,
#              A= None, W = None,
#              use_scipy = False,
#              device = "cuda" if torch.cuda.is_available() else "cpu",
#              dt = torch.float32):
#     """
#     Find centroids of an anisotropic power diagram. 
#     Inputs:
#     D - the spatial dimension
#     N - the number of grains
#     M - the number of quadrature points in each direction (M**D quadrature points in total)
#     X - the PyTorch tensor storing seed point positions
#     A - the PyTorch tensor storing the anistropy matrices of grains
#     W - the PyTorch tensor / numPy array storing the weights
#     use_scipy - whether W is passed as an array (optional, default is False)
#     """
#     if y_length == None:
#         y_length = x_length
    
#     if z_length == None and D == 3:
#         z_length = x_length
        
#     Y, PS = sample_grid(M, dim=D, x_length=x_length, y_length=y_length, z_length=z_length)
#     Y = Y.to(device,dtype=dt)
#     PS = PS.to(device,dtype=dt)
    
#     if A == None:
#         A = generate_identity_matrices(N,dim=D)
#     # FIX THIS
#     #if W == None:
#     #    W = np.zeros(N) if use_scipy else torch.zeros(N).to(device)
    
#     if use_scipy:
#         W = torch.from_numpy(W).to(device=device, dtype=dt)
    
#     w = LazyTensor(W.view(N,1,1))
    
#     y = LazyTensor(Y.view(1, M**D, D))  # (1,M**D, D)
#     x = LazyTensor(X.view(N, 1, D))  # (N, 1, D)
#     a = LazyTensor(A.view(N, 1, D * D))

#     D_ij = ((y - x) | a.matvecmult(y - x)) - w  # (N, M**D) symbolic LazyTensor

#     # Find which grain each pixel belongs to
#     grain_indices = D_ij.argmin(dim=0).ravel() # grain_indices[i] is the grain index of the i-th voxel
    
#     new_X0 = torch.bincount(grain_indices,Y[:,0],minlength=N)
#     new_X1 = torch.bincount(grain_indices,Y[:,1],minlength=N)
    
#     normalisation = torch.bincount(grain_indices,minlength=N)
#     if D == 3:
#         new_X2 = torch.bincount(grain_indices,Y[:,2],minlength=N)
#         return torch.stack([new_X0/normalisation, new_X1/normalisation,new_X2/normalisation],dim=1)
#     else:
#         return torch.stack([new_X0/normalisation, new_X1/normalisation],dim=1)

    