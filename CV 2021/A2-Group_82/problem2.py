"""
Wenhua Bao: 2512664
Zhenfan Song: 2864671
Kexin Wang: 2540047

"""
import numpy as np
import os
from PIL import Image
from scipy.spatial.distance import cdist
import pdb

#
# Task 1
#
def load_faces(path, ext=".pgm"):
    """Load faces into an array (N, M),
    where N is the number of face images and
    d is the dimensionality (height*width for greyscale).
    
    Hint: os.walk() supports recursive listing of files 
    and directories in a path
    
    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)
    
    Returns:
        x: (N, M) array
        hw: tuple with two elements (height, width)
    """
    

    img_list = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                a = Image.open(os.path.join(root, file))
                b = np.array(a)
                img_list.append(b)

    hw = b.shape
    x = np.asarray(img_list)
    x = x.reshape(len(img_list), hw[0]*hw[1])
    # pdb.set_trace()
    return x, hw

#
# Task 2
#

"""
This is a multiple-choice question
"""

class PCA(object):

    # choice of the method
    METHODS = {
                1: "SVD",
                2: "Eigendecomposition"
    }

    # choice of reasoning
    REASONING = {
                1: "it can be applied to any matrix and is more numerically stable",
                2: "it is more computationally efficient for our problem",
                3: "it allows to compute eigenvectors and eigenvalues of any matrix",
                4: "we can find the eigenvalues we need for our problem from the singular values",
                5: "we can find the singular values we need for our problem from the eigenvalues"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of PCA
            - the following integers provide the reasoning for your choice

        For example (made up):
            (2, 1, 5) means
            "I will use eigendecomposition because
                - we can apply it to any matrix
                - we need singular values which we can obtain from the eigenvalues"
        """

        return (1, 1, 2, 4)

#
# Task 3
#

def compute_pca(X):
    """PCA implementation
    
    Args:
        X: (N, M) an array with N M-dimensional features
    
    Returns:
        u: (M, N) bases with principal components
        lmb: (N, ) corresponding variance
    """

    N = len(X)
    X_mean = np.mean(X, axis=0)
    X_hat = X - X_mean
    U, S, vh = np.linalg.svd(X_hat)
    lmb = np.diagonal((1 / N) * np.dot(np.diag(S), np.diag(S)))
    # pdb.set_trace()
    return U, lmb

#
# Task 4
#

def basis(u, s, p = 0.5):
    """Return the minimum number of basis vectors
    from matrix U such that they account for at least p percent
    of total variance.
    
    Hint: Do the singular values really represent the variance?
    
    Args:
        u: (M, M) contains principal components.
        For example, i-th vector is u[:, i]
        s: (M, ) variance along the principal components.
    
    Returns:
        v: (M, D) contains M principal components from N
        containing at most p (percentile) of the variance.
    
    """

    N = len(u)
    for D in range(1, N):
        if np.sum(s[0:D]) >= p * np.sum(s[:N]):
            break

    v = u[:, :D]
    # pdb.set_trace()
    return v

#
# Task 5
#
def project(face_image, u):
    """Project face image to a number of principal
    components specified by num_components.
    
    Args:
        face_image: (N, ) vector (N=h*w) of the face
        u: (N,M) matrix containing M principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        image_out: (N, ) vector, projection of face_image on 
        principal components
    """

    image_out = np.dot(u,u.T@face_image)
    return image_out

#
# Task 6
#

"""
This is a multiple-choice question
"""
class NumberOfComponents(object):

    # choice of the method
    OBSERVATION = {
                1: "The more principal components we use, the sharper is the image",
                2: "The fewer principal components we use, the smaller is the re-projection error",
                3: "The first principal components mostly correspond to local features, e.g. nose, mouth, eyes",
                4: "The first principal components predominantly contain global structure, e.g. complete face",
                5: "The variations in the last principal components are perceptually insignificant; these bases can be neglected in the projection"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple describing you observations

        For example: (1, 3)
        """

        return (1, 4, 5)


#
# Task 7
#
def search(Y, x, u, top_n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        Y: (N, M) centered array with N d-dimensional features
        x: (1, M) image we would like to retrieve
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        top_n: integer, return top_n closest images in L2 sense.
    
    Returns:
        Y: (top_n, M)
    """

    new_Y = np.empty(Y.shape)
    for i in range(len(Y)):
        new_Y[i] = project(Y[i], u)
    Y_projected = new_Y.reshape(Y.shape)
    new_X = project(x,u)
    X_projected = new_X.reshape(1, len(new_X))
    dis = cdist(Y_projected , X_projected, 'euclidean').flatten()
    new_dis = np.argsort(dis)
    sorted_dis = Y[ new_dis].reshape(Y_projected.shape)

    return sorted_dis[:top_n,:]

#
# Task 8
#
def interpolate(x1, x2, u, N):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        x1: (1, M) array, the first image
        x2: (1, M) array, the second image
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        N: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate N equally-spaced points on a line
    
    Returns:
        Y: (N, M) interpolated results. The first dimension is in the index into corresponding
        image; Y[0] == project(x1, u); Y[-1] == project(x2, u)
    """
    p_x1 = project(x1, u)
    p_x2 = project(x2, u)
    a = np.linspace(p_x1,p_x2,N)

    return a
