import numpy as np
import matplotlib.pyplot as plt
import math
# Plot 2D points


def displaypoints2d(points):
  plt.figure(0)
  plt.plot(points[0,:],points[1,:], '.b')
  plt.xlabel('Screen X')
  plt.ylabel('Screen Y')


# Plot 3D points
def displaypoints3d(points):
  fig = plt.figure(1)
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(points[0,:], points[1,:], points[2,:], 'b')
  ax.set_xlabel("World X")
  ax.set_ylabel("World Y")
  ax.set_zlabel("World Z")


def cart2hom(points):
  """ Transforms from cartesian to homogeneous coordinates.

  Args:
    points: a np array of points in cartesian coordinates

  Returns:
    points_hom: a np array of points in homogeneous coordinates
  """

  #
  # You code here
  z=np.ones((1,points.shape[1]))
  hom=np.concatenate((points, z))

  return hom
  #
 


def hom2cart(points):
  """ Transforms from homogeneous to cartesian coordinates.

  Args:
    points: a np array of points in homogenous coordinates

  Returns:
    points_hom: a np array of points in cartesian coordinates
  """

  #
  # You code here
  cart=np.empty((points.shape[0]-1,points.shape[1]))
  for i in range(points.shape[1]):
      for j in range(cart.shape[0]):
          cart[j][i]=points[j][i]/points[-1][i]

  return cart
  #



def gettranslation(v):
  """ Returns translation matrix T in homogeneous coordinates for translation by v.

  Args:
    v: 3d translation vector

  Returns:
    T: translation matrix in homogeneous coordinates
  """

  #
  # You code here
  T=np.array([[1,0,0,v[0]],
              [0,1,0,v[1]],
              [0,0,1,v[2]],
              [0,0,0,1]])
  return T
  #


def getxrotation(d):
  """ Returns rotation matrix Rx in homogeneous coordinates for a rotation of d degrees around the x axis.

  Args:
    d: degrees of the rotation

  Returns:
    Rx: rotation matrix
  """

  #
  # You code here
  d=d*math.pi/180
  Rx=np.array([[1,0,0,0],
               [0,math.cos(d),-math.sin(d),0],
               [0,math.sin(d),math.cos(d),0],
               [0,0,0,1]])
  return Rx
  #


def getyrotation(d):
  """ Returns rotation matrix Ry in homogeneous coordinates for a rotation of d degrees around the y axis.

  Args:
    d: degrees of the rotation

  Returns:
    Ry: rotation matrix
  """

  #
  # You code here
  d=d*math.pi/180
  Ry=np.array([[math.cos(d),0,math.sin(d),0],
               [0,1,0,0],
               [-math.sin(d),0,math.cos(d),0],
               [0,0,0,1]])
  return Ry  
  #


def getzrotation(d):
  """ Returns rotation matrix Rz in homogeneous coordinates for a rotation of d degrees around the z axis.

  Args:
    d: degrees of the rotation

  Returns:
    Rz: rotation matrix
  """

  #
  # You code here
  d=d*math.pi/180
  Rz=np.array([[math.cos(d),-math.sin(d),0,0],
               [math.sin(d),math.cos(d),0,0],
               [0,0,1,0],
               [0,0,0,1]])
  return Rz 
  #


def getcentralprojection(principal, focal):
  """ Returns the (3 x 4) matrix L that projects homogeneous camera coordinates on homogeneous
  image coordinates depending on the principal point and focal length.
  
  Args:
    principal: the principal point, 2d vector
    focal: focal length

  Returns:
    L: central projection matrix
  """

  #
  # You code here
  L=np.array([[focal,0,principal[0],0],
              [0,focal,principal[1],0],
              [0,0,1,0]])
  return L
  
  #


def getfullprojection(T, Rx, Ry, Rz, L):
  """ Returns full projection matrix P and full extrinsic transformation matrix M.

  Args:
    T: translation matrix
    Rx: rotation matrix for rotation around the x-axis
    Ry: rotation matrix for rotation around the y-axis
    Rz: rotation matrix for rotation around the z-axis
    L: central projection matrix

  Returns:
    P: projection matrix
    M: matrix that summarizes extrinsic transformations
  """

  #
  # You code here
  M=Rz.dot(Rx).dot(Ry).dot(T)
  #check rotation
  #M=Rx.dot(Ry).dot(Ry).dot(T)
  #not commutative
  #check translation
  #M=T.dot(Rz).dot(Rx).dot(Ry)
  #bot commutative
  P=L.dot(M)
  return P,M
  #



def projectpoints(P, X):
  """ Apply full projection matrix P to 3D points X in cartesian coordinates.

  Args:
    P: projection matrix
    X: 3d points in cartesian coordinates

  Returns:
    x: 2d points in cartesian coordinates
  """

  #
  # You code here
  X_hom=cart2hom(X)
  x_hom=P.dot(X_hom)
  x=hom2cart(x_hom)
  return x
  #


def loadpoints():
  """ Load 2D points from obj2d.npy.

  Returns:
    x: np array of points loaded from obj2d.npy
  """

  #
  # You code here
  return np.load('data/obj2d.npy')
  #


def loadz():
  """ Load z-coordinates from zs.npy.

  Returns:
    z: np array containing the z-coordinates
  """

  #
  # You code here
  return np.load('data/zs.npy')
  #



def invertprojection(L, P2d, z):
  """
  Invert just the projection L of cartesian image coordinates P2d with z-coordinates z.

  Args:
    L: central projection matrix
    P2d: 2d image coordinates of the projected points
    z: z-components of the homogeneous image coordinates

  Returns:
    P3d: 3d cartesian camera coordinates of the points
  """

  #
  # You code here
  P3d=np.empty((3,P2d.shape[1]))
  P2d_hom=np.empty((P2d.shape[0]+1,P2d.shape[1]))
  for i in range(2):
      for j in range(P2d.shape[1]):
          P2d_hom[i][j]=z[0][j]*P2d[i][j]
  for i in range(P2d.shape[1]):
      P2d_hom[2][i]=z[0][i] 
  f=L[0][0]
  px=L[0][2]
  py=L[1][2]
  for i in range(P3d.shape[1]):      
      P3d[0][i]=(P2d_hom[0][i]-z[0][i]*px)/f
      P3d[1][i]=(P2d_hom[1][i]-z[0][i]*py)/f
  for i in range(P3d.shape[1]):
      P3d[2][i]=z[0][i]

  return P3d
  #


def inverttransformation(M, P3d):
  """ Invert just the model transformation in homogeneous coordinates
  for the 3D points P3d in cartesian coordinates.

  Args:
    M: matrix summarizing the extrinsic transformations
    P3d: 3d points in cartesian coordinates

  Returns:
    X: 3d points after the extrinsic transformations have been reverted
  """

  #
  # You code here
  P3d_hom=cart2hom(P3d)
  P=np.linalg.inv(M).dot(P3d_hom)
  return P
  #

def p3multiplecoice():
  '''
  Change the order of the transformations (translation and rotation).
  Check if they are commutative. Make a comment in your code.
  Return 0, 1 or 2:
  0: The transformations do not commute.
  1: Only rotations commute with each other.
  2: All transformations commute.
  '''

  return 0