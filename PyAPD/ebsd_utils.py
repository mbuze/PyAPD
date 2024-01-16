import torch
import numpy as np

#from .geometry_utils import *

def load_setup_from_EBSD_data_2D(file = "../../data/2D_basic_example/EBSD_example_2D_data.txt",
                                 seed_id = (1,2),
                                 volumes_id = (3),
                                 orientation_id = (4,5,6),
                                 normalise_matrices = True,
                                 angle_in_degrees = True,
                                 device = "cuda" if torch.cuda.is_available() else "cpu",
                                 dt = torch.float32):
    """ 
    Load geometric setup from 2D EBSD data. The data is located in `/data`.
    The file has to be such that each line corresponds to one grain.
    The default settings are for the deafult file where each line looks as follows:
    
    1 11.546 19.120 8.426e+002 21.38 13.78 103.5 973
    
    where
    1 is grain label
    (11.546,19.120) is the centroid of the grain (2D)
    8.426e+002 is the volume of the grain
    (21.38, 13.78, 103.5) defines a 2D ellipse (major axis, minor axis, rotation angle)
    973 is (presumably) the number of pixels (which determines the volume)
    
    Input:
    file - location of the file, should be of the form "../data/..." to load some data from the data folder
    (optional, default is "../data/2D_basic_example/EBSD_example_2D_data.txt" )
    seed_id - column ids storing location of each seed,
    volumes_id - column id storing volume of each seed,
    orientation_id - column ids storing orientation info of the grain (major axis, minor axis, rotation angle)
    normalise_matrices - whether to ensure that the anisotropy matrices have determinant equal to 1 or not (optional, default is True)
    device - "cuda" or "cpu" (optional default is "cuda" if it is available and "cpu" otherwise) 
    dt - data type used by Torch (optional default is torch.float32)
    """
    #EBSD = np.loadtxt("../data/EBSD_data.txt")
    EBSD = torch.from_numpy(np.loadtxt(file)).to(device=device, dtype=dt)
    X = EBSD[:,seed_id]
    TV = EBSD[:,volumes_id]
    if normalise_matrices:
        ratios = EBSD[:,orientation_id[0]] / EBSD[:,orientation_id[1]]
        major_axes = torch.sqrt(ratios)
        minor_axes = 1/major_axes
        thetas = EBSD[:,orientation_id[2]]*np.pi / 180 if angle_in_degrees else EBSD[:,orientation_id[2]]
    else:
        major_axes = EBSD[:,orientation_id[0]]
        minor_axes = EBSD[:,orientation_id[1]]
        thetas = EBSD[:,orientation_id[2]]*np.pi / 180 if angle_in_degrees else EBSD[:,orientation_id[2]]
    
    a11 = (1/major_axes**2)*torch.cos(thetas)**2 + (1/minor_axes**2)*torch.sin(thetas)**2
    a22 = (1/major_axes**2)*torch.sin(thetas)**2 + (1/minor_axes**2)*torch.cos(thetas)**2
    a12 = ((1/major_axes**2) - (1/minor_axes**2))*torch.cos(thetas)*torch.sin(thetas)
    a1 = torch.stack([a11, a12],dim=1)
    a2 = torch.stack([a12, a22],dim=1)
    A = torch.stack([a1,a2],dim=2)
    return X.contiguous(), A.contiguous(), TV.contiguous(), EBSD
    


