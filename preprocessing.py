#load data
import numpy as np
import argparse
from scipy.stats import linregress
from scipy.signal import firwin, butter, cheby1, lfiltic, lfilter, resample_poly
import cupy as cp

# FILTERING

def standardSVD(rawImages):
    # works for 2D x ensemble or 3D x ensemble
    numFrames = rawImages.shape[-1]
    flatImages = np.reshape(rawImages, (-1, numFrames), order='F')
    return np.linalg.svd(flatImages, full_matrices=False) # returns U, S, Vh

def standardSVDGPU(rawImages):
    rawImages = cp.array(rawImages)
    # works for 2D x ensemble or 3D x ensemble
    numFrames = rawImages.shape[-1]
    flatImages = cp.reshape(rawImages, (-1, numFrames), order='F')
    return cp.linalg.svd(flatImages, full_matrices=False) # returns U, S, Vh

def randomSVD(rawImages, tissueThreshold, d, iters):
    numFrames = rawImages.shape[-1]
    flatImages = np.reshape(rawImages, (-1, numFrames), order='F')
    k = tissueThreshold
    # k = expected singular value threshold for tissue
    _,t = flatImages.shape
    omega = np.random.randn(t, k+d)
    # project S onto a random lower dimensional matrix
    SPrime = flatImages@omega
    # Q = approximate orthonormal basis for S
    Q,_ = np.linalg.qr(SPrime)
    # power iterations to improve accuracy
    for i in range(iters):
        print(i)
        # amplifies upper singular values, suppresses lower ones
        # ensures Q more closesly aligns with top k singular values
        Qi,_ = np.linalg.qr(flatImages.conj().T@Q)
        Q,_ = np.linalg.qr(flatImages@Qi)
    A = Q.conj().T@flatImages
    tissueClutter = Q@A
    filteredImages = flatImages - tissueClutter

    filteredImages = np.reshape(filteredImages, rawImages.shape, order='F')
    return filteredImages

def randomSVDGPU(rawImages, tissueThreshold, d, iters):
    rawImages = cp.array(rawImages, dtype='complex64')
    numFrames = rawImages.shape[-1]
    flatImages = cp.reshape(rawImages, (-1, numFrames), order='F')
    k = tissueThreshold
    # k = expected singular value threshold for tissue
    _,t = flatImages.shape
    omega = cp.random.randn(t, k+d)
    # project S onto a random lower dimensional matrix
    SPrime = flatImages@omega
    # Q = approximate orthonormal basis for S
    Q,_ = cp.linalg.qr(SPrime)
    # power iterations to improve accuracy
    for i in range(iters):
        print(f'power iteration #{i+1}')
        # amplifies upper singular values, suppresses lower ones
        # ensures Q more closesly aligns with top k singular values
        Qi,_ = cp.linalg.qr(flatImages.conj().T@Q)
        Q,_ = cp.linalg.qr(flatImages@Qi)
    A = Q.conj().T@flatImages
    tissueClutter = Q@A
    filteredImages = flatImages - tissueClutter

    filteredImages = cp.reshape(filteredImages, rawImages.shape, order='F')
    return filteredImages.get()


# def adaptiveSVD(rawImages, tissueThreshold, noiseThreshold, blockSize, blockOverlap):

#     numFrames = rawImages.shape[-1]
#     PRF = 500
#     T = 1 / PRF

#     step = blockSize - blockOverlap

#     x, z, t = rawImages.shape

#     numBlocksX = (x - blockSize) // step + 1
#     numBlocksZ = (z - blockSize) // step + 1

#     filteredImages = np.zeros((x, z, t), dtype='complex128')
#     coverageMap = np.ones((x, z, t), dtype=int)

#     for i in range(0, numBlocksX, step):
#         for j in range(0, numBlocksZ, step):

#             coverageMap[i:i+blockSize, j:j+blockSize, :] += 1
#             block = rawImages[i:i+blockSize, j:j+blockSize, :]
#             flatBlock = np.reshape(block, (-1, numFrames), order='F')

#             (U, S, Vh) = np.linalg.svd(flatBlock,full_matrices=False)

#             logS = 10*np.log10(S/np.max(S))
#             second_derivative = np.diff(logS, n=2)
#             elbowIndex = np.argmax(second_derivative) + 1
#             thresholdIndex = np.argmax(logS >= tissueThreshold)

#             lowerThreshold = np.max(elbowIndex, thresholdIndex)

#             V = Vh.conj().T

#             v0 = V[:-1, :]
#             v1 = V[1:, :]
#             product = v1 * np.conj(v0)

#             num = np.sum(np.imag(product), axis=0)
#             den = np.sum(np.real(product), axis=0)
#             # mean angular frequency
#             omega = np.arctan2(num, den) / T
#             # mean doppler frequency
#             mean_freqs = np.abs(omega / (2 * np.pi))

#             fittingPoint = np.argmax(mean_freqs >= noiseThreshold)

#             slope, intercept, *_ = linregress(np.arange(fittingPoint, len(logS)), logS[fittingPoint:])
#             fittedSlope = slope * np.arange(len(logS)) + intercept
#             diff = np.abs(logS - fittedSlope)

#             diffThreshold = diff.mean() + 2*diff.std()
#             upperThreshold = len(diff) - 1 - np.argmax(diff[::-1] >= diffThreshold)

#             filteredBlock = reconstructSVD(U, S, Vh, lowerThreshold, upperThreshold, block.shape)
#             SSum = sum(S[tissueThreshold:noiseThreshold+1])
#             filteredImages[i:i+blockSize, j:j+blockSize, :] += filteredBlock/SSum

#     filteredImages /= coverageMap
#     filteredImages = np.transpose(filteredImages, (2, 0, 1))
#     return filteredImages


def adaptive_thresholding(S, Vh, tissueFreqThreshold, noiseFreqThreshold, gradientThreshold):

    PRF = 1000
    T = 1 / PRF

    logS = 20*np.log10(S/np.max(S)+1e-8)

    gradient = np.abs(np.gradient(logS))
    flatteningIndex = np.argmax(gradient[1:] <= gradientThreshold)

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

    thresholdIndex = np.argmax(mean_freqs >= tissueFreqThreshold)
    lowerThreshold = max(thresholdIndex, flatteningIndex)

    fittingPoint = np.argmax(mean_freqs >= noiseFreqThreshold)
    x_vals = np.arange(fittingPoint.item(), len(logS))
    y_vals = logS[fittingPoint:]

    slope, intercept, *_ = linregress(x_vals, y_vals)

    fittedSlope = slope * np.arange(len(logS)) + intercept
    diff = np.abs(logS - fittedSlope)

    # arbitrary threshold for point of deviation from linear trend
    diffThreshold =  2 * np.std(diff)  
    
    upperThreshold = len(diff) - 1 - np.argmax(diff[::-1] >= diffThreshold) 

    return lowerThreshold, upperThreshold

def blockwise_SVD_2D(rawImages, blockSize, blockOverlap, tissue_freq_thresh, noise_freq_thresh, gradient_thresh, gpu):

    step = blockSize - blockOverlap

    x, z, t = rawImages.shape

    filteredImages = np.zeros((x, z, t), dtype='complex128')
    coverageMap = np.zeros((x, z, t), dtype=int)

    for i in range(0, x - blockSize + 1, step):
        for j in range(0, z - blockSize + 1, step):
            block = rawImages[i:i+blockSize, j:j+blockSize, :]
            
            if gpu:
                (U, S, Vh) = standardSVDGPU(cp.array(block))
                U = U.get()
                S = S.get()
                Vh = Vh.get()
            else:
                (U, S, Vh) = standardSVD(block)

            lowerThreshold, upperThreshold = adaptive_thresholding(S, Vh, tissue_freq_thresh, noise_freq_thresh, gradient_thresh)

            filteredBlock = reconstructSVDGPU(U, S, Vh, lowerThreshold, upperThreshold, block.shape)
            SSum = sum(S[lowerThreshold:upperThreshold])
            filteredImages[i:i+blockSize, j:j+blockSize, :] += filteredBlock/SSum
            coverageMap[i:i+blockSize, j:j+blockSize, :] += 1

    filteredImages = np.where(coverageMap > 0, filteredImages / coverageMap, 0)
    return filteredImages

def blockwise_SVD_3D(rawImages, blockSize, blockOverlap, tissue_freq_thresh, noise_freq_thresh, gradient_thresh, gpu):

    step = blockSize - blockOverlap

    x, y, z, t = rawImages.shape

    filteredImages = np.zeros((x, y, z, t), dtype='complex128')
    coverageMap = np.zeros((x, y, z, t), dtype=int)


    for i in range(0, x - blockSize + 1, step):
        for j in range(0, y - blockSize + 1, step):
            for k in range(0, z - blockSize + 1, step):
                print(int(i/step), int(j/step), int(k/step))
                block = rawImages[i:i+blockSize, j:j+blockSize, k:k+blockSize, :]

                if gpu:
                    (U, S, Vh) = standardSVDGPU(cp.array(block))
                    U = U.get()
                    S = S.get()
                    Vh = Vh.get()
                else:
                    (U, S, Vh) = standardSVD(block)

                lowerThreshold, upperThreshold = adaptive_thresholding(S, Vh, tissue_freq_thresh, noise_freq_thresh, gradient_thresh)

                filteredBlock = reconstructSVDGPU(U, S, Vh, lowerThreshold, upperThreshold, block.shape)
                SSum = sum(S[lowerThreshold:upperThreshold])
                filteredImages[i:i+blockSize, j:j+blockSize, :] += filteredBlock/SSum
                coverageMap[i:i+blockSize, j:j+blockSize, :] += 1

        filteredImages = np.where(coverageMap > 0, filteredImages / coverageMap, 0)

    return filteredImages



def reconstructSVD(U, S, Vh, tissueThreshold, noiseThreshold, rawShape):
    S_filtered = S.copy()
    S_filtered[:tissueThreshold] = 0
    S_filtered[noiseThreshold:] = 0
    filteredImages = U@np.diag(S_filtered)@Vh
    filteredImages = np.reshape(filteredImages, rawShape, order='F')
    return filteredImages

def reconstructSVDGPU(U, S, Vh, tissueThreshold, noiseThreshold, rawShape):
    S_filtered = S
    S_filtered[0:tissueThreshold] = 0
    S_filtered[noiseThreshold:] = 0
    filteredImages = U@cp.diagflat(S_filtered)@Vh
    filteredImages = cp.reshape(filteredImages, rawShape, order='F')
    return filteredImages.get()

def IIR(data, order, fcutoff, initialization):
    #butterworth IIR filter

    if order==0:
        return data
    
    ensembleSize = data.shape[-1]

    # apply butterworth high-pass filter. 
    (b, a) = butter(order, fcutoff, 'highpass')

    # F = state matrix
    F = np.zeros((order, order))
    if order >= 2:
        for n in range(1, order):
            F[n-1, n] = 1
    F[order-1, :] = -a[:order]

    # g = impulse response vector
    g = b[:order] - b[order]*a[:order]
    g = np.reshape(g, (-1, 1))

    # q = observation vector
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
    
    # projects out the contribution of initial states
    if initialization == 'projection':
        coef = (np.eye(C.shape[0]) - B @ np.linalg.pinv(B.T @ B) @B.T) @C

    # models a step response from initial conditions
    elif initialization == 'step':
        L = np.zeros(ensembleSize)
        L[0] = 1
        coef = B @ np.linalg.pinv(np.eye(F.shape[0])-F)@ q @ L + C
    
    # assumes zero initial state
    elif initialization == 'zero':
        coef = C
    
    output = np.zeros_like(data)
    if len(data.shape) == 3:
        for m in range(data.shape[0]):
            for n in range(data.shape[1]):
                output[m, n, :] = coef @ data[m, n, :]
    else: 
        for m in range(data.shape[0]):
            for n in range(data.shape[1]):
                for o in range(data.shape[2]):
                    output[m, n, o, :] = coef @ data[m, n, o, :]

    return output

def regression(data, order): 
    # polynomial regression detrender. 

    # if order = 1 --> modeling signal as a0 + a1*t
    # solve for best-fit slope a2 and intercept a0, then compute trend line (least square)
    # trend(t) = a0 + a1t
    # subtract trend from original signal to remove slow, low-frequency components
    
    ensembleSize = data.shape[-1]
    dataRange = np.arange(1, ensembleSize+1)
    A = np.vstack([dataRange**n for n in range(order+1)])
    A2 = np.linalg.pinv(A@A.T)@A

    output = np.zeros_like(data)
    
    if len(data.shape) == 3:
        for i in range(data.shape[1]):
            for j in range(data.shape[0]):
                x = data[j, i, :]
                coeffs = A2 @ x
                trend = coeffs @ A
                output[j, i, :] = trend
    else:
        for k in range(data.shape[2]):
            for i in range(data.shape[1]):
                for j in range(data.shape[0]):
                    x = data[j, i, k, :]
                    coeffs = A2 @ x
                    trend = coeffs @ A
                    output[j, i, k, :] = trend

    return data - output

def butterworth(data, order, cutoff):
    # butterworth highpass filter
    # smooth, flat passband
    (b, a) = butter(order, cutoff, 'highpass')
    return highpass(data, b, a, order)

def chbyshv(data, order, cutoff, rolloff):
    # chebyshev highpass filter
    # ripple in pass band, sharp cutoff
    (b, a) = cheby1(order, rolloff, cutoff, 'highpass')
    return highpass(data, b, a, order)

def FIR(data, order, cutoff):
    b = firwin(order, cutoff, pass_zero=False)
    return lfilter(b, [1.0], data, axis=2) 

def highpass(data, b, a, order):
    shape = data.shape
    zi = np.zeros((order, shape[0], shape[1]))
    output = np.zeros_like(data)
    for depth in range(shape[0]):
        for ensemble in range(shape[1]):
            zi_vec = lfiltic(b, a, y=np.zeros(order), x=data[depth, ensemble, 0]*np.ones(order))
            zi[:, depth, ensemble] = zi_vec
            output[depth, ensemble, :],_ = lfilter(b, a, data[depth, ensemble, :], zi=zi_vec)
    return output

def velEst(data, fc, fprf):
    # autocorrelation velocity estimation
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
    # threshold to remove low velocities
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

def rand_downsample(rawImages, nBases, lowerThreshold, upperThreshold, gpu):
    numFrames = rawImages.shape[-1]
    flatImages = np.reshape(rawImages, (-1, numFrames), order='F')
    pxs, t = flatImages.shape

    rand_rows = np.random.permutation(pxs)

    split_rows = np.array_split(rand_rows, nBases)

    filteredIm = np.zeros((pxs, t), dtype='complex128')

    # thresholdMap = np.zeros(pxs)

    for i in range(nBases):
        print(i)
        row_indices = split_rows[i]
        submatrix = flatImages[row_indices, :]

        if gpu:
            (U, S, Vh) = standardSVDGPU(cp.array(submatrix))
            U = U.get()
            S = S.get()
            Vh = Vh.get()
        else:
            (U, S, Vh) = standardSVD(submatrix)



        filteredSubmatrix = reconstructSVD(U, S, Vh, lowerThreshold, upperThreshold, submatrix.shape)
        filteredIm[row_indices, :] = filteredSubmatrix
        # thresholdMap[row_indices] = lowerThreshold.get()
    filteredIm = np.reshape(filteredIm, rawImages.shape, order='F')
    # thresholdMap = np.reshape(thresholdMap, (rawImages.shape[0], rawImages.shape[1]), order='F')
    return filteredIm

def decimate(rawImages, D):
    return rawImages[::D, :, :]

def interpolate(rawImages, ID):
    return resample_poly(rawImages, up=ID, down=1, axis=1)


def hilbert_cp(x, axis=-1):
    N = x.shape[axis]
    Xf = cp.fft.fft(x, axis=axis)

    h = cp.zeros(N, dtype=cp.float32)
    if N % 2 == 0:
        h[0] = 1
        h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2

    shape = [1]*x.ndim
    shape[axis] = N
    h = h.reshape(shape)

    x_hilbert = cp.fft.ifft(Xf * h, axis=axis)
    return x_hilbert