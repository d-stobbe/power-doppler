import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap, Normalize

def makeBmode(images):
  # sum IQ data
  averageBmode = np.mean(images, axis=2)
  # envelope detection
  absBmode = np.abs(averageBmode)
  # logarithmic decomp
  log_env = 20*np.log10(absBmode/np.max(absBmode)+1e-8)
  log_env[log_env < -40] = -40
  BmodeArray = log_env
  return BmodeArray

def makeFlow(Filteredimages, minMag, sigma, dynrange):

  pwr = np.abs(Filteredimages)**2
  flowArray = np.mean(pwr, axis=-1)
  flowArray = 10*np.log10(flowArray/np.max(flowArray)+1e-8)
  flowArray[flowArray < -dynrange] = -dynrange

  cmap = plt.get_cmap('hot')
  norm = plt.Normalize(-dynrange, 0)

  rgbaFlow = cmap(norm(flowArray))

  #thresholding for overlaying power doppler image onto bmode image
  minMag = np.percentile(flowArray, minMag)
  transparencyMap = (flowArray>=minMag)

  rgbaFlow[:, :, -1] = transparencyMap.astype(float)

  return rgbaFlow

def plotBlood(powerIntensityMat, maxMag, sigma):
    rgbaFlow = makeFlow(powerIntensityMat, maxMag, 0, sigma)
    plt.imshow(rgbaFlow)
    plt.show()

def plotPowerDoppler(powerIntensityMat, bmodeArray, maxMag, minMag, sigma):
    plt.imshow(bmodeArray, cmap='grey')
    rgbaFlow = makeFlow(powerIntensityMat, maxMag, minMag, sigma)
    plt.imshow(rgbaFlow)
    plt.show()