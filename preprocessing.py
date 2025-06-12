#load data
from scipy.io import loadmat
import numpy as np
from scipy.stats import linregress
from scipy.signal import firwin

def standardSVD(rawImages):
    numFrames = rawImages.shape[-1]
    flatImages = np.reshape(rawImages, (-1, numFrames), order='F')
    return np.linalg.svd(flatImages, full_matrices=False) # returns U, S, Vh

def randomSVD(rawImages, tissueThreshold, d, iters):
    numFrames = rawImages.shape[-1]
    flatImages = np.reshape(rawImages, (-1, numFrames), order='F')
    k = tissueThreshold
    # k = expected singular value threshold for tissue
    numFrames = rawImages.shape[0]
    _,t = flatImages.shape
    omega = np.random.randn(t, k+d)
    # project S onto a random lower dimensional matrix
    SPrime = flatImages@omega
    # Q = approximate orthonormal basis for S
    Q,_ = np.linalg.qr(SPrime)
    # power iterations to improve accuracy
    for _ in range(iters):
        # amplifies upper singular values, suppresses lower ones
        # ensures Q more closesly aligns with top k singular values
        Qi,_ = np.linalg.qr(flatImages.conj().T@Q)
        Q,_ = np.linalg.qr(flatImages@Qi)
    A = Q.conj().T@flatImages
    tissueClutter = Q@A
    filteredImages = flatImages - tissueClutter

    filteredImages = np.reshape(filteredImages, rawImages.shape, order='F')
    return filteredImages

def adaptiveSVD(rawImages, tissueThreshold, noiseThreshold, blockSize, blockOverlap):

    numFrames = rawImages.shape[-1]
    PRF = 500
    T = 1 / PRF

    step = blockSize - blockOverlap

    x, z, t = rawImages.shape

    numBlocksX = (x - blockSize) // step + 1
    numBlocksZ = (z - blockSize) // step + 1

    filteredImages = np.zeros((x, z, t), dtype='complex128')
    coverageMap = np.ones((x, z, t), dtype=int)

    for i in range(0, numBlocksX, step):
        for j in range(0, numBlocksZ, step):

            coverageMap[i:i+blockSize, j:j+blockSize, :] += 1
            block = rawImages[i:i+blockSize, j:j+blockSize, :]
            flatBlock = np.reshape(block, (-1, numFrames), order='F')

            (U, S, Vh) = np.linalg.svd(flatBlock,full_matrices=False)

            logS = 20*np.log10(S/np.max(S))
            second_derivative = np.diff(logS, n=2)
            elbowIndex = np.argmax(second_derivative) + 1
            thresholdIndex = np.argmax(logS >= tissueThreshold)

            lowerThreshold = np.max(elbowIndex, thresholdIndex)

            V = Vh.conj().T

            v0 = V[:-1, :]
            v1 = V[1:, :]
            product = v1 * np.conj(v0)

            num = np.sum(np.imag(product), axis=0)
            den = np.sum(np.real(product), axis=0)
            # mean angular frequency
            omega = np.arctan2(num, den) / T
            # mean doppler frequency
            mean_freqs = np.abs(omega / (2 * np.pi))

            fittingPoint = np.argmax(mean_freqs >= noiseThreshold)

            slope, intercept, *_ = linregress(np.arange(fittingPoint, len(logS)), logS[fittingPoint:])
            fittedSlope = slope * np.arange(len(logS)) + intercept
            diff = np.abs(logS - fittedSlope)

            diffThreshold = diff.mean() + 2*diff.std()
            upperThreshold = len(diff) - 1 - np.argmax(diff[::-1] >= diffThreshold)

            filteredBlock = reconstructSVD(U, S, Vh, lowerThreshold, upperThreshold, block.shape)
            SSum = sum(S[tissueThreshold:noiseThreshold+1])
            filteredImages[i:i+blockSize, j:j+blockSize, :] += filteredBlock/SSum

    filteredImages /= coverageMap
    filteredImages = np.transpose(filteredImages, (2, 0, 1))
    return filteredImages

def reconstructSVD(U, S, Vh, tissueThreshold, noiseThreshold, rawShape):
    S_filtered = S.copy()
    S_filtered[0:tissueThreshold+1] = 0
    S_filtered[noiseThreshold:len(S_filtered)] = 0
    filteredImages = U@np.diag(S_filtered)@Vh
    filteredImages = np.reshape(filteredImages, rawShape, order='F')
    return filteredImages

# def randomDownsample(rawImages, nBases):
#     numFrames = rawImages.shape[-1]
#     flatImages = np.reshape(rawImages, (-1, numFrames), order='F')
#     nBases = 16
#     xy, t = flatImages.shape

#     rand_rows = np.random.permutation(xy)

#     split_rows = np.array_split(rand_rows, nBases)

#     filteredIm = np.zeros((xy, t), dtype='complex128')

#     for i in range(nBases):
#         row_indices = split_rows[i]
#         submatrix = flatImages[row_indices, :]
#         filtedSubmatrix = submatrix # do svd here
#         filteredIm[row_indices, :] = filtedSubmatrix # reconstruction
#     return filteredIm

def lowpassFilter(numtaps, cutoff):
    return firwin(numtaps, cutoff, pass_zero=True)

def highpassFilter(numtaps, cutoff):
    return firwin(numtaps, cutoff, pass_zero=True)

def bandpassFilter(numtaps, cutoff):
    return firwin(numtaps, cutoff, pass_zero=False)

