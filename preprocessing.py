#load data
import numpy as np
from scipy.stats import linregress
from scipy.signal import firwin, butter, cheby1, lfiltic, lfilter

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

def IIR(data, order, fcutoff, initialization):
    if order==0:
        return data
    
    ensembleSize = data.shape[-1]

    (b, a) = butter(order, fcutoff, 'highpass')
    F = np.zeros((order, order))
    if order >= 2:
        for n in range(1, order):
            F[n-1, n] = 1
    F[order-1, :] = -a[:order]
    g = b[:order] - b[order]*a[:order]
    g = np.reshape(g, (-1, 1))
    q = np.zeros((order, 1))
    q[-1, 0] = 1

    B = np.zeros((ensembleSize, order))
    F_power = np.eye(order)
    for n in range(ensembleSize):
        B[n, :] = (g.T@F_power).flatten()
        F_power = F_power@F
    
    C = np.zeros((ensembleSize, ensembleSize))
    C[0, 0] = b[order]
    F_power = np.eye(order)
    for n in range(ensembleSize - 1):
        C[n+1, 0] = (g.T @ F_power @ q).item()
        F_power = F_power @ F
    
    for n in range(1, ensembleSize):
        C[n:, n] = C[0:ensembleSize-n, 0]
    
    if initialization == 'projection':
        coef = (np.eye(C.shape[0]) - B @ np.linalg.pinv(B.T @ B) @B.T) @C

    elif initialization == 'step':
        L = np.zeros((1, ensembleSize))
        L[0, 0] = 1
        coef = B @ np.linalg.pinv(np.eye(F.shape[0])-F)@ q @ L + C
    elif initialization == 'zero':
        coef = C
    
    output = np.zeros_like(data)

    for m in range(0, data.shape[0]):
        for n in range(0, data.shape[1]):
            output[m, n, :] = coef @ data[m, n, :]

    return output

def regression(data, order): 

    ensembleSize = data.shape[-1]
    dataRange = np.arange(1, ensembleSize+1)
    A = np.vstack([dataRange**n for n in range(order+1)])
    A2 = np.linalg.pinv(A@A.T)@A
    output = np.zeros_like(data)

    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            x = data[j, i, :]
            coeffs = A2 @ x
            trend = coeffs @ A
            output[j, i, :] = trend
    
    return data - output

def butterworth(data, order, cutoff):
    (b, a) = butter(order, cutoff, 'highpass')
    return highpass(data, b, a, order)

def chbychv(data, order, cutoff, rolloff):
    (b, a) = cheby1(order, rolloff, cutoff, 'highpass')
    return highpass(data, b, a, order)

def FIR(data, order, cutoff):
    b = firwin(order+1, cutoff, pass_zero=False)
    return lfilter(b, [1.0], data, axis=2) 

def highpass(data, b, a, order):
    shape = data.shape
    zi = np.zeros((order, shape[0], shape[1]))
    output = np.zeros_like(data)
    for depth in range(shape[0]):
        for ensemble in range(shape[1]):
            zi_vec = lfiltic(b, a, y=np.zeros(order), x=data[depth, ensemble, 0]*np.ones(order))
            zi[:, depth, ensemble] = zi_vec
            output[depth, ensemble, :],_ = lfilter(b, a, data[depth, ensemble, :], zi_vec)
    return output

def velEst(data, fc, fprf):
    c = 1540
    vAxial = np.zeros(data.shape[:-1])
    
    data_shifted_1 = data[:, :, :-1]
    data_shifted_2 = data[:, :, 1:]

    autocorr = data_shifted_2 * data_shifted_1.conj()

    sum_imag = np.sum(np.imag(autocorr), axis=2)
    sum_real = np.sum(np.real(autocorr), axis=2)

    vAxial = (c * fprf) / (4 * np.pi *fc * 1e6) * np.arctan2(sum_imag, sum_real)
    return vAxial

def velFilt(data, fc, fprf, velThresh):
    c = 1540
    vAxial = velEst(data, fc, fprf)
    maxVel = (c * fprf) / (4 * fc * 1e6)
    vAxialFilt = vAxial
    vAxialFilt = vAxialFilt*(np.abs(vAxialFilt)>maxVel*velThresh)
    return vAxialFilt

def tissueFilt(data, powerPre, tissueThresh):
    return data * (powerPre < (1-tissueThresh))

def clutterFilt(data, powerPost, clutterThresh):
    return data * (powerPost > (clutterThresh))