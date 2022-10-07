from copy import deepcopy
import numpy as np
from PIL import Image
from scipy.special import binom
from scipy.ndimage import convolve


def loadimg(path):
    """ Load image file

    Args:
        path: path to image file
    Returns:
        image as (H, W) np.array normalized to [0, 1]
    """

    #
    # You code here
    img=Image.open(path)
    return np.asarray(img)/255
    #
    

def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (H, W) np.array
    """

    #
    # You code here
    h,w=fsize[0],fsize[1]
    x=np.arange(-h/2+0.5,h/2)
    y=np.arange(-w/2+0.5,w/2)
    X,Y=np.meshgrid(x, y,sparse=True)
    g=np.exp(-(X**2+Y**2)/(2*sigma**2))
    return g/np.sum(g)        
    #


def binomial2d(fsize):
    """ Create a 2D binomial filter

    Args:
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* binomial filter as (H, W) np.array
    """

    #
    # You code here
    h,w=fsize[0],fsize[1]
    X=np.empty((1,h))
    Y=np.empty((w,1))
    for i in range(h):
        X[0][i]=binom(h-1,i) 
    for i in range(w):
        Y[i][0]=binom(w-1,i) 
    X=X
    Y=Y
    bi=Y.dot(X)

    return bi/np.sum(bi)
    #
    




def downsample2(img, f):
    """ Downsample image by a factor of 2
    Filter with Gaussian filter then take every other row/column

    Args:
        img: image to downsample
        f: 2d filter kernel
    Returns:
        downsampled image as (H, W) np.array
    """

    #
    # You code here
    img1=convolve(img,f)
    return img1[::2,::2]
    #   

def upsample2(img, f):
    """ Upsample image by factor of 2

    Args:
        img: image to upsample
        f: 2d filter kernel
    Returns:
        upsampled image as (H, W) np.array
    """

    #
    # You code here
    H,W=img.shape
    img1=np.zeros((2*H,2*W))
    for i in range(H):
        for j in range(W):
            img1[2*i][2*j]=img[i][j]
    
    img2=convolve(img1,f)
    return 4*img2
    #

def gaussianpyramid(img, nlevel, f):
    """ Build Gaussian pyramid from image

    Args:
        img: input image for Gaussian pyramid
        nlevel: number of pyramid levels
        f: 2d filter kernel
    Returns:
        Gaussian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """

    #
    # You code here
    pyr=[]
    img1=deepcopy(img)
    for i in range(nlevel):
        pyr.append(img1)
        img1=downsample2(img1, f)
    return pyr   
    #


def laplacianpyramid(gpyramid, f):
    """ Build Laplacian pyramid from Gaussian pyramid

    Args:
        gpyramid: Gaussian pyramid
        f: 2d filter kernel
    Returns:
        Laplacian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """

    #
    # You code here
    pyr=[]
    gpyramid1=deepcopy(gpyramid[::-1])
    pyr.append(gpyramid1[0])
    for i in range(len(gpyramid1)-1):         
        pyr.append(gpyramid1[i+1]-upsample2(gpyramid1[i], f))
    return pyr[::-1]
    
    #

def reconstructimage(lpyramid, f):
    """ Reconstruct image from Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        f: 2d filter kernel
    Returns:
        Reconstructed image as (H, W) np.array clipped to [0, 1]
    """

    #
    # You code here
    l=deepcopy(lpyramid[::-1])
    G=l[0]
    for i in range(1,len(l)):
        img= upsample2(G, f)+l[i]
        G=img
    return img
    #


def amplifyhighfreq(lpyramid, l0_factor=4.0, l1_factor=4.0):
    """ Amplify frequencies of the finest two layers of the Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        l0_factor: amplification factor for the finest pyramid level
        l1_factor: amplification factor for the second finest pyramid level
    Returns:
        Amplified Laplacian pyramid, data format like input pyramid
    """

    #
    # You code here
    l=list()
    l.append(l0_factor*lpyramid[0])
    l.append(l1_factor*lpyramid[1])
    for i in range(2,len(lpyramid)):
        l.append(lpyramid[i])
    return l
    #


def createcompositeimage(pyramid):
    """ Create composite image from image pyramid
    Arrange from finest to coarsest image left to right, pad images with
    zeros on the bottom to match the hight of the finest pyramid level.
    Normalize each pyramid level individually before adding to the composite image

    Args:
        pyramid: image pyramid
    Returns:
        composite image as (H, W) np.array
    """

    #
    # You code here
    H=pyramid[0].shape[0]
    W=0
    for p in pyramid:
        W+=p.shape[1]
    img=np.empty((H,W))
    offset=0
    for p in pyramid:
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                if p[i][j]<0:
                    img[i][j+offset]=0
                elif p[i][j]>1:
                    img[i][j+offset]=1
                else:
                    img[i][j+offset]=p[i][j]
        offset+=p.shape[1]
        
    return img
    
    #



