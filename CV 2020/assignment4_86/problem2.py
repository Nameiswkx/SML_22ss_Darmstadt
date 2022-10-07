from functools import partial
import numpy as np
from PIL import Image
from scipy import interpolate
from scipy.ndimage import convolve


####################
# Provided functions
####################


conv2d = partial(convolve, mode="mirror")


def gauss2d(fsize, sigma):
    """ Create a 2D Gaussian filter

    Args:
        fsize: (w, h) dimensions of the filter
        sigma: width of the Gaussian filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def downsample(img, fsize=(5, 5), sigma=1.4):
    """
    Downsampling an image by a factor of 2

    Args:
        img: image as (h, w) np.array
        fsize and sigma: parameters for Gaussian smoothing
                         to apply before the subsampling
    Returns:
        downsampled image as (h/2, w/2) np.array
    """
    g_k = gauss2d(fsize, sigma)
    img = conv2d(img, g_k)
    return img[::2, ::2]


def gaussian_pyramid(img, nlevels=3, fsize=(5, 5), sigma=1.4):
    """ Build Gaussian pyramid from image

    Args:
        img: input image for Gaussian pyramid
        nlevel: number of pyramid levels
        fsize: gaussian kernel size
        sigma: sigma of gaussian kernel

    Returns:
        Gaussian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    pyramid = [img]
    for i in range(0, nlevels - 1):
        pyramid.append(downsample(pyramid[i], fsize, sigma))

    return pyramid


def resize(arr, shape):
    """ Resize an image to target shape

    Args:
        arr: image as (h, w) np.array
        shape: target size (h', w') as tuple

    Returns:
        resized image as (h', w') np.array
    """
    return np.array(Image.fromarray(arr).resize(shape[::-1]))


######################
# Basic Lucas-Kanade #
######################


def compute_derivatives(im1, im2):
    """Compute dx, dy and dt derivatives

    Args:
        im1: first image as (h, w) np.array
        im2: second image as (h, w) np.array

    Returns:
        Ix, Iy, It: derivatives of im1 w.r.t. x, y and t
                    as (h, w) np.array
    """

    #
    # You code here
    fy= np.array([[-0.5, 0, 0.5]])
    Iy=convolve(im1,fy,mode='mirror')
    fx=fy.T
    Ix=convolve(im2,fx,mode='mirror')
    It=im2-im1
    return Ix,Iy,It
    #

def compute_motion(Ix, Iy, It, patch_size=15):
    """Computes one iteration of optical flow estimation.

    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t each as (h, w) np.array
        patch_size: specifies the side of the square region R in Eq. (1)
    Returns:
        u: optical flow in x direction as (h, w) np.array
        v: optical flow in y direction as (h, w) np.array
    """

    #
    # You code here
    half=int((patch_size-1)/2)
    Ix1=np.pad(Ix,half,'reflect')
    Iy1=np.pad(Iy,half,'reflect')
    It1=np.pad(It,half,'reflect')
    h,w=Ix.shape
    u=np.empty((h,w))
    v=np.empty((h,w))
    
    for i in range(half,h+half):
        for j in range(half,w+half):

            ix=Ix1[i-half:i+half+1,j-half:j+half+1].flatten()
            iy=Iy1[i-half:i+half+1,j-half:j+half+1].flatten()
            it=It1[i-half:i+half+1,j-half:j+half+1].flatten()
            A=np.array([[np.sum(ix*ix),np.sum(ix*iy)],
                        [np.sum(ix*iy),np.sum(iy*iy)]])
            B=np.array([[np.sum(ix*it)],
                        [np.sum(iy*it)]])
            u0,v0=-np.linalg.pinv(A)@B
            u[i-half,j-half]=u0
            v[i-half,j-half]=v0
    return u,v
             

    #

        

def warp(im, u, v):
    """Warping of a given image using provided optical flow.

    Args:
        im: input image as (h, w) np.array
        u, v: optical flow in x and y direction each as (h, w) np.array

    Returns:
        im_warp: warped image as (h, w) np.array
    """

    #
    # You code here
    h,w=im.shape
    points=np.empty((h*w,2))
    
    for i in range(h):
        for j in range(w):
            points[i*w+j][0]=i-u[i,j]
            points[i*w+j][1]=j-v[i,j]
    

    values=im.flatten()
    xi, yi= np.mgrid[0:h, 0:w] 
 
    im_warp=interpolate.griddata(points,values,(xi,yi),method='linear',fill_value=0)
    return im_warp
    
    #
    

  

def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade.
    Args:
        im1, im2: Images as (h, w) np.array

    Returns:
        Cost as float scalar
    """             

    a=im1-im2
    cost=np.mean([k**2 for k in a])

    return cost
    #
    

###############################
# Coarse-to-fine Lucas-Kanade #
###############################

def coarse_to_fine(pyramid1, pyramid2, n_iter=10):
    """Implementation of coarse-to-fine strategy
    for optical flow estimation.

    Args:
        pyramid1, pyramid2: Gaussian pyramids corresponding to
                            im1 and im2, in fine to coarse order
        n_iter: number of refinement iterations

    Returns:
        u: OF in x direction as np.array
        v: OF in y direction as np.array
    """

    #
    # You code here
    l=len(pyramid1)
    u=np.zeros(pyramid1[-1].shape)
    v=np.zeros(pyramid1[-1].shape)

    
    for j in range(n_iter):
        u=(0.5**(l-1))*resize(u, pyramid1[-1].shape)
        v=(0.5**(l-1))*resize(v, pyramid1[-1].shape)

        im1=pyramid1[-1]
        im2=pyramid2[-1] 
        for i in range(1,l+1):                
            im1=warp(pyramid1[-i], u, v)
            im2=pyramid2[-i]
            Ix, Iy, It=compute_derivatives(im1, im2)
            u_,v_=compute_motion(Ix, Iy, It)
            u+=u_
            v+=v_
            
            if(i<l):
                shape=(2*u.shape[0],2*u.shape[1])
                u=2*resize(u, shape)
                v=2*resize(v, shape)
        
    return u,v
    
    #
