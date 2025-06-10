from ipywidgets import IntRangeSlider, IntSlider, FloatSlider, interactive_output, Output, Layout, HBox, VBox, Dropdown, Button, ToggleButton, ToggleButtons
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import standardSVD, randomSVD, adaptiveSVD, reconstructSVD
from visualization import makeBmode, makeFlow, plotBlood, plotPowerDoppler


def power2D(rawImages):
    
    bloodOut = Output()
    powerDopplerOut = Output()

    shape = rawImages.shape
    maxEnsemble = shape[-1]

    class Random:
        def __init__(self):
            pass
        def filter(self, images, minMag, maxMag, sigma, k, d, iters, ensemble):
            self.k = k
            self.d = d
            self.iters = iters
            self.ensemble = ensemble
            self.filteredImages = randomSVD(images, self.k, self.d, self.iters)
            self.update(minMag, maxMag, sigma)
        def update(self, minMag, maxMag, sigma):
            self.minMag = minMag
            self.maxMag = maxMag
            self.sigma = sigma

    class Adaptive:
        def __init__(self):
            pass
        def filter(self, images, minMag, maxMag, sigma, tissueThreshold, noiseThreshold, blockSize, blockOverlap, ensemble):
            self.ensemble = ensemble
            self.tissueThreshold = tissueThreshold
            self.noiseThreshold = noiseThreshold
            self.blockSize = blockSize
            self.blockOverlap = blockOverlap
            self.filteredImages = adaptiveSVD(images, self.tissueThreshold, self.noiseThreshold, self.blockSize, self.blockOverlap)
            self.update(minMag, maxMag, sigma)
        def update(self, minMag, maxMag, sigma):
            self.minMag = minMag
            self.maxMag = maxMag
            self.sigma = sigma

    class Standard:
        def __init__(self):
            pass
        def filter(self, images, thresholds, minMag, maxMag, sigma, ensemble):
            (self.U, self.S, self.Vh) = standardSVD(images)
            self.ensemble = ensemble
            self.reconstruct(thresholds, minMag, maxMag, sigma, images.shape)
        def reconstruct(self, thresholds, minMag, maxMag, sigma, shape):
            self.filteredImages = reconstructSVD(self.U, self.S, self.Vh, thresholds[0], thresholds[1], shape)
            self.thresholds = thresholds
            self.minMag = minMag
            self.maxMag = maxMag
            self.sigma = sigma
            self.shape = shape

    ensembleSize = IntRangeSlider(
        value=[0, maxEnsemble],
        min=0,
        max=maxEnsemble,
        step=1,
        description='ensemble:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
    )

    thresholdSlider = IntRangeSlider(
        value=[0, maxEnsemble],
        min=0,
        max=maxEnsemble,
        step=1,
        description='SVD range:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style={'description_width': '130px'},
        layout=Layout(width='400px')
    )

    maxPowerSlider = FloatSlider(
        min=0, max=100, step=0.5, value=99.5,
        description='Max Power:',
        continuous_update=False,
        style={'description_width': '130px'},
        layout=Layout(width='400px')
    )

    minPowerSlider = FloatSlider(
        min=0, max=100, step=0.5, value=0,
        description='Min Power:',
        continuous_update=False,
        # style={'description_width': '130px'},
        layout=Layout(padding='50px 0px 0px 0px')
    )

    sigmaSlider = FloatSlider(
        min=0, max=10, step=0.1, value=4.0,
        description='Sigma :',
        continuous_update=False,
        style={'description_width': '130px'},
        layout=Layout(width='400px')
    )

    svdSelector = ToggleButtons(
        options=['Standard','Random', 'Adaptive'],
        description='SVD Mode:',
        button_style='',
        tooltips=['Randomized decomposition', 'Basic SVD', 'Adaptive approach'],
        continuous_update=False
    )

    randomK = IntSlider(
        min=0, max=256, step=1, value=256,
        description='k:',
        continuous_update=False,
        style={'description_width': '130px'}
    )

    randomIters = IntSlider(
        min=0, max=256, step=1, value=256,
        description='Power Iters:',
        continuous_update=False,
        style={'description_width': '130px'}
    )

    randomD = IntSlider(
        min=0, max=256, step=1, value=256,
        description='d:',
        continuous_update=False,
        style={'description_width': '130px'}
    )

    adaptiveBlockSize = IntSlider(
        min=0, max=256, step=1, value=256,
        description='Block Size:',
        continuous_update=False,
        style={'description_width': '130px'}
    )

    adaptiveBlockOverlap = IntSlider(
        min=0, max=256, step=1, value=256,
        description='Block Overlap:',
        continuous_update=False,
        style={'description_width': '130px'}
    )

    adaptiveTissueThreshold = IntSlider(
        min=0, max=256, step=1, value=256,
        description='Tissue Threshold:',
        continuous_update=False,
        style={'description_width': '130px'}
    )

    adaptiveNoiseThreshold = IntSlider(
        min=0, max=256, step=1, value=256,
        description='Noise Threshold:',
        continuous_update=False,
        style={'description_width': '130px'}
    )

    filterButton = Button(
        description='Filter',
        disabled=False,
        button_style='',
        tooltip='Filter',
        icon='check'
    )

    loadButton = ToggleButton(
        description='load prev',
        disabled=False,
        button_style='',
        tooltip='Filter',
        icon='check', 
        value=False
    )

    random = Random()
    adaptive = Adaptive()
    standard = Standard()

    def changeSize(change):
        thresholdSlider.unobserve(sliderChange, names='value')
        thresholdSlider.min = 0
        thresholdSlider.max = change['new'][1] - change['new'][0]
        thresholdSlider.value = [0, change['new'][1] - change['new'][0]]
        thresholdSlider.observe(sliderChange, names='value')

    def toggleMode(change):
        bloodOut.clear_output(wait=False)
        powerDopplerOut.clear_output(wait=False)
        loadButton.value = False
        if change['new'] == 'Standard':
            standInputs.layout.display = 'flex'
            randInputs.layout.display = 'none'
            adaptiveInputs.layout.display = 'none'
        elif change['new'] == 'Random':
            standInputs.layout.display = 'none'
            randInputs.layout.display = 'flex'
            adaptiveInputs.layout.display = 'none'
        else:
            standInputs.layout.display = 'none'
            randInputs.layout.display = 'none'
            adaptiveInputs.layout.display = 'flex'

    def loadPrev(change):
        svdMode = svdSelector.value
        if change['new']:
            if svdMode == 'Standard':
                thresholdSlider.value = standard.thresholds
                ensembleSize.value = standard.ensemble
                minPowerSlider.value = standard.minMag
                maxPowerSlider.value = standard.maxMag
                sigmaSlider.value = standard.sigma
                displayPlots(standard.maxMag, standard.minMag, standard.sigma, thresholds=standard.thresholds)
            elif svdMode == 'Random':
                ensembleSize.value = random.ensemble
                randomK.value = random.k
                randomD.value = random.d
                randomIters.value = random.iters
                minPowerSlider.value = random.minMag
                maxPowerSlider.value = random.maxMag
                sigmaSlider.value = random.sigma
                displayPlots(random.maxMag, random.minMag, random.sigma)
            else:
                ensembleSize.value = adaptive.ensemble
                adaptiveBlockSize.value = adaptive.blockSize
                adaptiveBlockOverlap.value = adaptive.blockOverlap
                adaptiveTissueThreshold.value = adaptive.tissueThreshold
                adaptiveNoiseThreshold.value = adaptive.noiseThreshold
                minPowerSlider.value = adaptive.minMag
                maxPowerSlider.value = adaptive.maxMag
                sigmaSlider.value = adaptive.sigma
                displayPlots(adaptive.maxMag, adaptive.minMag, adaptive.sigma)
            # not working properly -- nested VBox
            for child in svdParams.children:
                child.disabled = True
        else:
            for child in svdParams.children:
                child.disabled = False
            


    def filter(b):
        if not loadButton.value:
            ensemble = rawImages[:, :, ensembleSize.value[0]:ensembleSize.value[1]]
            svdMode = svdSelector.value
            if svdMode == 'Standard':
                standard.filter(ensemble, thresholdSlider.value, minPowerSlider.value, maxPowerSlider.value, sigmaSlider.value, ensembleSize.value)
                displayPlots(standard.maxMag, standard.minMag, standard.sigma, thresholds=standard.thresholds)
            elif svdMode == 'Random':
                random.filter(ensemble, minPowerSlider.value, maxPowerSlider.value, sigmaSlider.value, randomK.value, randomD.value, randomIters.value, ensembleSize.value)
                displayPlots(random.maxMag, random.minMag, random.sigma)
            else:
                adaptive.filter(ensemble, minPowerSlider.value, maxPowerSlider.value, sigmaSlider.value, adaptiveTissueThreshold.value, adaptiveNoiseThreshold.value, adaptiveBlockSize.value, adaptiveBlockOverlap.value, ensembleSize.value)
                displayPlots(adaptive.maxMag, adaptive.minMag, adaptive.sigma)
            loadButton.value = True

    def displayPlots(maxMag, minMag, sigma, **kwargs):
        ensemble = rawImages[:, :, ensembleSize.value[0]:ensembleSize.value[1]]
        if svdSelector.value == 'Standard':
            thresholds = kwargs.get('thresholds')
            standard.reconstruct(thresholds, minMag, maxMag, sigma, standard.shape)
            filteredImages = standard.filteredImages
        elif svdSelector.value == 'Random':
            random.update(minMag, maxMag, sigma)
            filteredImages = random.filteredImages
        else:
            adaptive.update(minMag, maxMag, sigma)
            filteredImages = adaptive.filteredImages
            
        bloodOut.clear_output(wait=True)
        with bloodOut:
            plt.figure(figsize=(5, 5))
            plt.title('Filtered Blood Signal')
            plt.imshow(makeFlow(filteredImages, maxMag, 0, sigma))
            plt.show()
        powerDopplerOut.clear_output(wait=True)
        with powerDopplerOut:
            bmodeArray = makeBmode(ensemble)
            plt.figure(figsize=(5, 5))
            plt.title('Power Doppler')
            plt.imshow(bmodeArray, cmap='grey')
            rgbaFlow = makeFlow(filteredImages, maxMag, minMag, sigma)
            plt.imshow(rgbaFlow)
            plt.show()

    def sliderChange(change):
        if change['name'] == 'value' and change['type'] == 'change':
            displayPlots(
                maxMag=maxPowerSlider.value,
                minMag=minPowerSlider.value,
                sigma=sigmaSlider.value,
                thresholds=thresholdSlider.value
            )

    svdSelector.observe(toggleMode, names='value')
    ensembleSize.observe(changeSize, names='value')
    filterButton.on_click(filter)
    loadButton.observe(loadPrev, names='value')
    maxPowerSlider.observe(sliderChange, names='value')
    minPowerSlider.observe(sliderChange, names='value')
    sigmaSlider.observe(sliderChange, names='value')
    thresholdSlider.observe(sliderChange, names='value')


    randInputs = VBox([randomK, randomIters, randomD])

    adaptiveInputs = VBox([adaptiveBlockSize, adaptiveBlockOverlap, adaptiveTissueThreshold, adaptiveNoiseThreshold])

    svdParams = VBox([ensembleSize, randInputs, adaptiveInputs])

    standInputs = VBox([thresholdSlider])

    reconParams = VBox([standInputs, sigmaSlider, maxPowerSlider, minPowerSlider])

    standInputs.layout.display = 'flex'
    randInputs.layout.display = 'none'
    adaptiveInputs.layout.display = 'none'

    display(HBox([VBox([svdSelector, svdParams, HBox([filterButton, loadButton]), reconParams]), bloodOut, powerDopplerOut]))

def power3D():
    return