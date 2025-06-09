import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def makeBmode(images):
  # sum IQ data
  averageBmode = np.mean(images, axis=2)
  # envelope detection
  absBmode = np.abs(averageBmode)
  # logarithmic decomp
  log_env = 20*np.log10(absBmode/np.max(absBmode))
  log_env[log_env < -40] = -40
  BmodeArray = log_env
  return BmodeArray

def makeFlow(Filteredimages, maxMag, minMag, sigma):
  # sum the powers
  powerFlow = np.abs(Filteredimages)**2
  averageFlow = powerFlow.mean(axis=2)
  # averageFlow = 10*np.log10(averageFlow/np.max(averageFlow))
  maxMag = np.percentile(averageFlow, maxMag)
  # assume power above maxMag is tissue
  averageFlow[averageFlow > maxMag] = 0

  averageFlow = gaussian_filter(averageFlow, sigma=sigma, mode='constant')

  flowArray = averageFlow

  cmap = plt.get_cmap('autumn')
  norm = plt.Normalize(np.min(flowArray), np.max(flowArray))

  rgbaFlow = cmap(norm(flowArray))
  #thresholding to retrieve relevant blood signal
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