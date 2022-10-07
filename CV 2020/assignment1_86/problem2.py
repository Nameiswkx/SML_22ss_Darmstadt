import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
def loaddata(path):
    """ Load bayerdata from file

    Args:
        Path of the .npy file
    Returns:
        Bayer data as numpy array (H,W)
    """

    #
    # You code here
    return np.load(path)
    #

def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """

    #
    # You code here
    H,W=bayerdata.shape[:2]
    r,g,b=[np.zeros(bayerdata.shape) for i in range(3)]
    for i in range(int(H/2)):
        for j in range(int(W/2)):
            r[2*i][2*j+1]=bayerdata[2*i][2*j+1]
            b[2*i+1][2*j]=bayerdata[2*i+1][2*j]
            g[2*i][2*j]=bayerdata[2*i][2*j]
            g[2*i+1][2*j+1]=bayerdata[2*i+1][2*j+1]
    #
    return r,g,b

def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    H,W=r.shape[:2]
    img=np.zeros((H,W,3))
    for i in range(int(H/2)):
        for j in range(int(W/2)):
            img[2*i][2*j+1][0]=r[2*i][2*j+1]
            img[2*i+1][2*j][2]=b[2*i+1][2*j]
            img[2*i][2*j][1]=g[2*i][2*j]
            img[2*i+1][2*j+1][1]=g[2*i+1][2*j+1] 
    return img
    #

def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """

    #
    # You code here
    H,W=r.shape[:2]
    img=np.zeros((H,W,3))

    
    b_r = np.array([[1/4, 0, 1/4],
                    [0, 0, 0],
                    [1/4, 0, 1/4]
                    ])
    
    b_g = np.array([[0, 1/4, 0],
                    [1/4, 0, 1/4],
                    [0, 1/4, 0]])    
    r_b=b_r
    r_g=b_g
    #g1 left top
    g1_r=np.array([[0, 0, 0],
                   [1/2, 0, 1/2],
                   [0, 0, 0]])
    
    g1_b=np.array([[0, 1/2, 0],
                   [0, 0, 0],
                   [0, 1/2, 0]])
    #g2 right bottom
    g2_r=g1_b
    g2_b=g1_r
      
    img_r_g=convolve(g, r_g,mode='reflect')
    img_r_b=convolve(b, r_b,mode='reflect')
    img_b_r=convolve(r, b_r,mode='reflect')
    img_b_g=convolve(g, b_g,mode='reflect')
    img_g1_r=convolve(r, g1_r,mode='reflect')
    img_g1_b=convolve(b, g1_b,mode='reflect')
    img_g2_r=convolve(r, g2_r,mode='reflect')
    img_g2_b=convolve(b, g2_b,mode='reflect')
    
    
    for i in range(int(H/2)):
        for j in range(int(W/2)):
            #r
            img[2*i][2*j+1][0]=r[2*i][2*j+1]
            img[2*i][2*j+1][1]=img_r_g[2*i][2*j+1]
            img[2*i][2*j+1][2]=img_r_b[2*i][2*j+1]
            #b
            img[2*i+1][2*j][2]=b[2*i+1][2*j]
            img[2*i+1][2*j][0]=img_b_r[2*i+1][2*j]
            img[2*i+1][2*j][1]=img_b_g[2*i+1][2*j]
            #g1
            img[2*i][2*j][1]=g[2*i][2*j]
            img[2*i][2*j][0]=img_g1_r[2*i][2*j]
            img[2*i][2*j][2]=img_g1_b[2*i][2*j]
            #g2
            img[2*i+1][2*j+1][1]=g[2*i+1][2*j+1]     
            img[2*i+1][2*j+1][0]=img_g2_r[2*i+1][2*j+1]
            img[2*i+1][2*j+1][2]=img_g2_b[2*i+1][2*j+1]
            

    return img
    #
