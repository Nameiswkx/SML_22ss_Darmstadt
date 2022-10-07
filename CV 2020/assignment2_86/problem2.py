import numpy as np
import os
from PIL import Image


def load_faces(path, ext=".pgm"):
    """Load faces into an array (N, H, W),
    where N is the number of face images and
    H, W are height and width of the images.
    
    Hint: os.walk() supports recursive listing of files 
    and directories in a path
    
    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)
    
    Returns:
        imgs: (N, H, W) numpy array
    """
    
    #
    # You code here
    imgs=[]
    for root, dirs, files in os.walk('./data/yale_faces', topdown=False):
       for name in files:
           if name.endswith(ext):
               imgs.append(np.array(Image.open(os.path.join(root, name))))
    return np.array(imgs)
    #


def vectorize_images(imgs):
    """Turns an  array (N, H, W),
    where N is the number of face images and
    H, W are height and width of the images into
    an (N, M) array where M=H*W is the image dimension.
    
    Args:
        imgs: (N, H, W) numpy array
    
    Returns:
        x: (N, M) numpy array
    """
    
    #
    # You code here
    N,H,W=imgs.shape
    M=H*W
    x=np.zeros((N,M))
    for n in range(N):
        for i in range(H):
            for j in range(W):
                x[n,i*W+j]=imgs[n,i,j]
    return x
  
    #


def compute_pca(X):
    """PCA implementation
    
    Args:
        X: (N, M) an numpy array with N M-dimensional features
    
    Returns:
        mean_face: (M,) numpy array representing the mean face
        u: (M, M) numpy array, bases with D principal components
        cumul_var: (N, ) numpy array, corresponding cumulative variance
    """

    #
    # You code here
    N,M=X.shape
    mean_face=np.mean(X,axis=0)
    face_n=X-mean_face
    cov = np.cov(face_n, rowvar = False)
    eigval, u = np.linalg.eig(np.mat(cov)) 
    idx=np.argsort(-eigval)
    eigval=eigval[idx].real
    u=u[:,idx].real
    cum_var = np.cumsum(eigval)
    return mean_face,np.array(u),np.array(cum_var[:N])
    #



def basis(u, cumul_var, p = 0.5):
    """Return the minimum number of basis vectors 
    from matrix U such that they account for at least p percent
    of total variance.
    
    Hint: Do the singular values really represent the variance?
    
    Args:
        u: (M, M) numpy array containing principal components.
        For example, i'th vector is u[:, i]
        cumul_var: (N, ) numpy array, variance along the principal components.
    
    Returns:
        v: (M, D) numpy array, contains M principal components from N
        containing at most p (percentile) of the variance.
    
    """
    
    #
    # You code here
    cumul_exp=cumul_var/cumul_var[-1]
    for i in range(len(cumul_exp)):
        if cumul_exp[i]>=p:
            return np.array(u[:,0:i])
    #


def compute_coefficients(face_image, mean_face, u):
    """Computes the coefficients of the face image with respect to
    the principal components u after projection.
    
    Args:
        face_image: (M, ) numpy array (M=h*w) of the face image a vector
        mean_face: (M, ) numpy array, mean face as a vector
        u: (M, D) numpy array containing D principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        a: (D, ) numpy array, containing the coefficients
    """
    
    #
    # You code here
    a=u.T.dot(face_image-mean_face)
    return np.array(a.T)
    #


def reconstruct_image(a, mean_face, u):
    """Reconstructs the face image with respect to
    the first D principal components u.
    
    Args:
        a: (D, ) numpy array containings the image coefficients w.r.t
        the principal components u
        mean_face: (M, ) numpy array, mean face as a vector
        u: (M, D) numpy array containing D principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        image_out: (M, ) numpy array, projected vector of face_image on 
        principal components
    """
    
    #
    # You code here
    return u.dot(a)+mean_face
    #


def compute_similarity(Y, x, u, mean_face):
    """Compute the similarity of an image x to the images in Y
    based on the cosine similarity.

    Args:
        Y: (N, M) numpy array with N M-dimensional features
        x: (M, ) image we would like to retrieve
        u: (M, D) bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector

    Returns:
        sim: (N, ) numpy array containing the cosine similarity values
    """

    #
    # You code here
    N=Y.shape[0]
    W=u.shape[1]
    a_Y=np.empty((N,W))
    for i in range(N):
        a_Y[i]=compute_coefficients(Y[i], mean_face, u)
    a_x=compute_coefficients(x, mean_face, u)
    sim=np.empty((N))
    for i in range(N):
        sim[i]=a_x.dot(a_Y[i])/(np.linalg.norm(a_x)*np.linalg.norm(a_Y[i]))
    return sim
        
    #


def search(Y, x, u, mean_face, top_n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        Y: (N, M) numpy array with N M-dimensional features
        x: (M, ) numpy array, image we would like to retrieve
        u: (M, D) numpy arrray, bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector
        top_n: integer, return top_n closest images in L2 sense.
    
    Returns:
        Y: (top_n, M) numpy array containing the top_n most similar images
        sorted by similarity
    """

    #
    # You code here
    sim=compute_similarity(Y, x, u, mean_face)
    sim_sorted,y_sorted=zip(*sorted(zip(sim,Y),reverse=True))
    y_sorted=np.array(y_sorted)
    return y_sorted[:top_n,:]
    #


def interpolate(x1, x2, u, mean_face, n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        x1: (M, ) numpy array, the first image
        x2: (M, ) numpy array, the second image
        u: (M, D) numpy array, bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector
        n: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate n equally-spaced points on a line
    
    Returns:
        Y: (n, M) numpy arrray, interpolated results.
        The first dimension is in the index into corresponding
        image; Y[0] == project(x1, u); Y[-1] == project(x2, u)
    """

    #
    # You code here
    W=u.shape[1]
    M=x1.shape[0]
    a_x1=np.empty((W,1))
    a_x2=np.empty((W,1))
    a_x1=compute_coefficients(x1, mean_face, u)
    a_x2=compute_coefficients(x2, mean_face, u)    
    a=np.linspace(a_x1, a_x2, num=n)
    Y=np.empty((n,M))
    for i in range(n):
        Y[i]=reconstruct_image(a[i], mean_face, u)

    return Y
    #




