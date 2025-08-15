# import numpy as np
from scipy.io import loadmat, savemat
import preprocessing
from visualization import makeFlow
import matplotlib.pyplot as plt



def PD_SVD(images, tissue_thresh, noise_thresh, dynrange, gpu):
    if gpu:
        U, S, Vh = preprocessing.standardSVDGPU(images)
        filtered_im = preprocessing.reconstructSVDGPU(U, S, Vh, tissue_thresh, noise_thresh, images.shape)
    else:
        U, S, Vh = preprocessing.standardSVD(images)
        filtered_im = preprocessing.reconstructSVD(U, S, Vh, tissue_thresh, noise_thresh, images.shape)
    if len(images.shape) == 3:
        pd = makeFlow(filtered_im, 0, 0, dynrange)
        plt.imshow(pd)
        plt.show()
    savemat(f'filtererd_im.mat', {'image': filtered_im})

# USING RANDOMIZED SVD IN https://ieeexplore.ieee.org/document/7845720
#'Accelerated Singular Value-Based Ultrasound Blood Flow Clutter Filtering With Randomized 
# Singular Value Decomposition and Randomized Spatial Downsampling'
def PD_rSVD(images, k, d, iters, dynrange, gpu):
    if gpu:
        filtered_im = preprocessing.randomSVDGPU(images, k, d, iters)
    else:
        filtered_im = preprocessing.randomSVD(images, k, d, iters)
    if len(images.shape) == 3:
        pd = makeFlow(filtered_im, 0, 0, dynrange)
        plt.imshow(pd)
        plt.show()
    savemat(f'filtererd_im.mat', {'image': filtered_im})

# USING SAME PROCESS AS https://pubmed.ncbi.nlm.nih.gov/27608455/
# 'Ultrasound Small Vessel Imaging With Block-Wise Adaptive Local Clutter Filtering' 

# adaptive filtering depends on characteristics of singular value plot and mean doppler freq of singular values
def PD_aSVD(images, tissue_thresh, noise_thresh, grad_thresh, blockSize, blockOverlap, dynrange, gpu):
    if len(images.shape) == 3:
        filtered_im = preprocessing.blockwise_SVD_2D(images, blockSize, blockOverlap, tissue_thresh, noise_thresh, grad_thresh, gpu)
    else:
        filtered_im = preprocessing.blockwise_SVD_3D(images, blockSize, blockOverlap, tissue_thresh, noise_thresh, grad_thresh, gpu)
    if len(images.shape) == 3:
        pd = makeFlow(filtered_im, 0, 0, dynrange)
        plt.imshow(pd)
        plt.show()
    savemat(f'filtererd_im.mat', {'image': filtered_im})

#NOTE: Before doing adaptive thresholding, you need to look at the 

# USING SAME PROCESS AS THE ADAPTIVE METHOD SUGGESTED IN 
# USING SVD ON RANDOMIZED SUBSETS IN https://ieeexplore.ieee.org/document/7845720
#'Accelerated Singular Value-Based Ultrasound Blood Flow Clutter Filtering With Randomized 
# Singular Value Decomposition and Randomized Spatial Downsampling'
def PD_rand_downsample(images, nBases, tissue_thresh, noise_thresh, dynrange, gpu):
    if gpu:
        filtered_im = preprocessing.rand_downsample(images, nBases, tissue_thresh, noise_thresh, gpu)
    else:
        filtered_im = preprocessing.rand_downsample(images, nBases, tissue_thresh, noise_thresh, gpu)
    if len(images.shape) == 3:
        pd = makeFlow(filtered_im, 0, 0, dynrange)
        plt.imshow(pd)
        plt.show()
    savemat(f'filtererd_im.mat', {'image': filtered_im})

## LOAD IMAGES

images = loadmat('test_data.mat')['testData']

## CALL PD METHOD

