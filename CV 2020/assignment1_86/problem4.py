import math
import numpy as np
from scipy import ndimage


def gauss2d(sigma, fsize):
  """
  Args:
    sigma: width of the Gaussian filter
    fsize: dimensions of the filter

  Returns:
    g: *normalized* Gaussian filter
  """

  #
  # You code here
  g=np.empty((fsize,1))
  n=int((fsize-1)/2)
  for i,x in zip(range(fsize),range(-n,n+1)):
      g[i]=(1/(math.sqrt(2*math.pi)*sigma))*math.exp(-x*x/(2*sigma*sigma))
  return g/g.sum()
  #


def createfilters():
  """
  Returns:
    fx, fy: filters as described in the problem assignment
  """
  
  #
  # You code here
  gx=gauss2d(0.9, 3)
  dx=0.5*np.array([[1,0,-1]])
  fx=gx.dot(dx)
  fy=fx.T
  return fx,fy
  #


def filterimage(I, fx, fy):
  """ Filter the image with the filters fx, fy.
  You may use the ndimage.convolve scipy-function.

  Args:
    I: a (H,W) numpy array storing image data
    fx, fy: filters

  Returns:
    Ix, Iy: images filtered by fx and fy respectively
  """

  #
  # You code here
  Ix=ndimage.convolve(I, fx)
  Iy=ndimage.convolve(I, fy)
  return Ix,Iy
  #
  
def detectedges(Ix, Iy, thr):
  """ Detects edges by applying a threshold on the image gradient magnitude.

  Args:
    Ix, Iy: filtered images
    thr: the threshold value

  Returns:
    edges: (H,W) array that contains the magnitude of the image gradient at edges and 0 otherwise
  """

  #
  # You code here
  #We chose the threshold of 0.1, because the main edges in the building is shown and the other edges like the trees and reflections are ignored
  H,W=Ix.shape
  edges=np.zeros(Ix.shape)
  for i in range(H):
      for j in range(W):
          mag=(Ix[i][j]**2+Iy[i][j]**2)**0.5
          if mag>thr:
              edges[i][j]=mag
  return edges
       
  #

def nonmaxsupp(edges, Ix, Iy):
  """ Performs non-maximum suppression on an edge map.

  Args:
    edges: edge map containing the magnitude of the image gradient at edges and 0 otherwise
    Ix, Iy: filtered images

  Returns:
    edges2: edge map where non-maximum edges are suppressed
  """

  #theta = np.arctan(Iy/ Ix)*180/np.pi
  edges2=np.zeros(edges.shape)
  theta=np.zeros(edges.shape)
  H,W=edges.shape  
  for i in range(H):
      for j in range(W):
          if Ix[i][j]==0:
              if Iy[i][j]>0:
                  theta[i][j]=90 
              else: theta[i][j]=-90 
          else:
              theta[i][j] = np.arctan(Iy[i][j]/ Ix[i][j])*180/np.pi
  
  # handle top-to-bottom edges: theta in [-90, -67.5] or (67.5, 90]
  
  # You code here
  for i in range(1,H-1):
      for j in range(1,W-1):
          a=1
          b=1
          if(-90<=theta[i][j]<=-67.5) or (67.5<theta[i][j]<=90) :
              a=edges[i-1][j]
              b=edges[i+1][j]
  # handle left-to-right edges: theta in (-22.5, 22.5]

  # You code here
          elif(-22.5<theta[i][j]<=22.5):
              a=edges[i][j+1]
              b=edges[i][j-1]

  # handle bottomleft-to-topright edges: theta in (22.5, 67.5]
  
  # Your code here
          elif(22.5<theta[i][j]<=67.5):
              a=edges[i-1][j+1]
              b=edges[i+1][j-1]

  

  # handle topleft-to-bottomright edges: theta in [-67.5, -22.5]
  # Your code here
          elif(-67.5<=theta[i][j]<=-22.5):
              a=edges[i+1][j+1]
              b=edges[i-1][j-1]
          
          if((edges[i][j]>=a) and (edges[i][j]>=b)):
              edges2[i][j]=edges[i][j]

  return edges2

