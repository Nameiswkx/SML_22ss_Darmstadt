import numpy as np
import matplotlib.pyplot as plt

def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    plt.imshow(img)
    plt.show()
    #


def save_as_npy(path, img):
    """ Save the image array as a .npy file:

    Args:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    np.save(path,img)
    #


def load_npy(path):
    """ Load and return the .npy file:

    Args:
        Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    return np.load(path)
    #


def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image:

    Args:
        Loaded image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """

    #
    # You code here
    H,W=img.shape[:2]
    img1=np.empty(img.shape)
    for i in range(H):
        for j in range(W):
            img1[i][j]=img[i][W-j-1]
    return img1

            

def display_images(img1, img2):
    """ display the normal and the mirrored image in one plot:

    Args:
        Two image numpy arrays
    """

    #
    # You code here
    plt.subplot(2,1,1)
    plt.imshow(img1)
    plt.subplot(2,1,2)
    plt.imshow(img2)
    plt.show()
    #
