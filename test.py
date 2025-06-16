from scipy.io import loadmat
from visualization import makeBmode
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize

from preprocessing import IIR, regression, butterworth, chbychv, FIR, velFilt, tissueFilt, clutterFilt

#acquire iq data
IQ = loadmat('testdata.mat')['IQ']

IQFilt = IIR(IQ, 1, 0.6, 'projection')

powerPreFilt = np.sum(np.abs(IQ)**2, axis=2)
powerPreFilt/=np.max(powerPreFilt)
powerPostFilt = np.sum(np.abs(IQFilt)**2, axis=2)
powerPostFilt/=np.max(powerPostFilt)

maxMag = 0.1

bmodeArray = np.log10(10+np.abs(IQ[:, :, 0]))
bmodeArray = bmodeArray - np.min(bmodeArray)
bmodeArray = (bmodeArray/np.max(np.abs(bmodeArray))*127/128-1)*maxMag

magnitude = powerPostFilt
magnitude[magnitude > maxMag] == maxMag

bmodeArray[magnitude > (maxMag / 40)] = magnitude[magnitude > (maxMag / 40)]
ImageArray = bmodeArray

PowerMap = np.zeros((256, 3))
PowerMap[0:128, :] = np.tile(np.linspace(0, 1, 128)[:, np.newaxis], (1, 3))
PowerMap[128:256, 0] = np.sqrt(np.log(np.arange(3, 131))) / np.sqrt(np.log(80))
PowerMap[131:256, 1] = np.sqrt(np.linspace(0, 132, 125)) / np.sqrt(128)
PowerMap = np.clip(PowerMap, 0, 1)

custom_cmap = ListedColormap(PowerMap, )

plt.imshow(ImageArray, cmap=custom_cmap, aspect='auto', norm=Normalize(vmin=-0.1, vmax=0.1))
plt.colorbar()
plt.show()

#IIR
#regression
#butterworth
#chebychev      --> IQFilt
#eigen (svd)
#FIR

# get power using IQ (prefilt)
# get power using IQFilt (postfilt)

# vel thresholding
# tissue thresholding (prefilt)
# smoothing post filt (use different methods)
# clutter thresholding (post filt)

# colormap/plotting?

