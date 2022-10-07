"""
Wenhua Bao: 2512664
Zhenfan Song: 2864671
Kexin Wang: 2540047

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def load_data(path):
    '''
    Load data from folder data, face images are in the folder facial_images, face features are in the folder facial_features.
    

    Args:
        path: path of folder data

    Returns:
        imgs: list of face images as numpy arrays 
        feats: list of facial features as numpy arrays 
    '''

    imgs = []
    feats = []
    for filename in os.listdir(path+'/facial_features'):
        feat = plt.imread(path+'/facial_features'+'/'+filename)
        feats.append(feat)
    for filename in os.listdir(path+'/'+'facial_images'):
        img = plt.imread(path+'/facial_images'+'/'+filename)
        imgs.append(img)
    return imgs, feats


def gaussian_kernel(fsize, sigma):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of Gaussian kernel

    Returns:
        The Gaussian kernel
    '''

    gk = np.empty((fsize, fsize))
    for i in range(gk.shape[0]):
        for j in range(gk.shape[1]):
            gk[i,j] = 1 / (2 * np.pi * sigma ** 2) * \
                      np.exp(-((i - np.median(range(fsize))) ** 2 + (j - np.median(range(fsize))) ** 2)
                               / (2 * (sigma ** 2)))

    return gk / np.sum(abs(gk))

def downsample_x2(x, factor=2):
    '''
    Downsampling an image by a factor of 2

    Args:
        x: image as numpy array (H * W)

    Returns:
        downsampled image as numpy array (H/2 * W/2)
    '''
    downsample = np.empty((int(np.ceil(x.shape[0]/2)), int(np.ceil(x.shape[1]/2))))
    for i in range(downsample.shape[0]):
        for j in range( downsample.shape[1]):
            downsample[i, j] = x[i*2, j*2]

    return downsample


def gaussian_pyramid(img, nlevels, fsize, sigma):
    '''
    A Gaussian pyramid is constructed by combining a Gaussian kernel and downsampling.
    Tips: use scipy.signal.convolve2d for filtering image.

    Args:
        img: face image as numpy array (H * W)
        nlevels: number of levels of Gaussian pyramid, in this assignment we will use 3 levels
        fsize: Gaussian kernel size, in this assignment we will define 5
        sigma: sigma of Gaussian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of Gaussian downsampled images, it should be 3 * H * W
    '''
    GP = []
    current_img = img.copy()
    GP.append(current_img)
    for i in range(nlevels-1):
        if i !=0:
            current_img = convolve2d(img, gaussian_kernel(fsize, sigma),  mode='same')
            current_img = downsample_x2(current_img, )
            GP.append(current_img)

    return GP

def template_distance(v1, v2):
    '''
    Calculates the distance between the two vectors to find a match.
    Browse the course slides for distance measurement methods to implement this function.
    Tips: 
        - Before doing this, let's take a look at the multiple choice questions that follow. 
        - You may need to implement these distance measurement methods to compare which is better.

    Args:
        v1: vector 1
        v2: vector 2

    Returns:
        Distance
    '''
    #SSD
    distance = np.linalg.norm(v1-v2)
    return distance


def sliding_window(img, feat, step=1):
    ''' 
    A sliding window for matching features to windows with SSDs. When a match is found it returns to its location.
    
    Args:
        img: face image as numpy array (H * W)
        feat: facial feature as numpy array (H * W)
        step: stride size to move the window, default is 1
    Returns:
        min_score: distance between feat and window
    '''

    #min_score = None
    dis = []
    img_h, img_w = img.shape
    feat_h, feat_w = feat.shape
    row_max = img_h-feat_h
    col_max = img_w-feat_w

    for i in range(0, row_max+1, step):
        for j in range(0, col_max+1, step):
            new_img = img[i:i + feat_h, j:j + feat_w]
            dis.append(template_distance(new_img, feat))
    if len(dis)>0:
        min_score = min(dis)
    else:
        min_score=-1


    return min_score


class Distance(object):

    # choice of the method
    METHODS = {1: 'Dot Product', 2: 'SSD Matching'}

    # choice of reasoning
    REASONING = {
        1: 'it is more computationally efficient',
        2: 'it is less sensitive to changes in brightness.',
        3: 'it is more robust to additive Gaussian noise',
        4: 'it can be implemented with convolution',
        5: 'All of the above are correct.'
    }

    def answer(self):
        '''Provide your answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of distance.
            - the following integers provide the reasoning for your choice.
        Note that you have to implement your choice in function template_distance

        For example (made up):
            (1, 1) means
            'I will use Dot Product because it is more computationally efficient.'
        '''

        return (2, 5)  # TODO


def find_matching_with_scale(imgs, feats):
    ''' 
    Find face images and facial features that match the scales 
    
    Args:
        imgs: list of face images as numpy arrays
        feats: list of facial features as numpy arrays 
    Returns:
        match: all the found face images and facial features that match the scales: N * (score, g_im, feat)
        score: minimum score between face image and facial feature
        g_im: face image with corresponding scale
        feat: facial feature
    '''
    match = []
    (score, g_im, feat) = (None, None, None)
    for feat in feats:
        template_mach = []
        scores = []
        for img in imgs:
            GP = gaussian_pyramid(img, 3,5,1.4)
            for g in GP:
                score = sliding_window(g, feat)
                if score >= 0:
                    template_mach.append((score, g, feat))
                    scores.append(score)
        match.append(template_mach[np.argmin(scores)])

    return match
