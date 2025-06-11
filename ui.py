from ipywidgets import IntRangeSlider, IntSlider, FloatSlider, Output, Layout, HBox, VBox, Dropdown, Button, ToggleButton, ToggleButtons, Checkbox
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

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

def power3D(data, spacing, origin):
    mag = np.abs(data)
    log_env = 20*np.log10(mag/(np.max(mag)+1e-8))
    log_env[log_env < -40] = -40
    bmodeArray = log_env
    bmodeArray = (log_env+40)/40
    # bmodeArray /= 2
    grid = pv.ImageData()
    grid.dimensions = bmodeArray.shape
    grid.spacing = spacing
    grid.origin = origin
    grid.point_data['intensity'] = bmodeArray.flatten(order='F')

    init_x = bmodeArray.shape[0] // 2
    init_y = bmodeArray.shape[1] // 2
    init_z = bmodeArray.shape[2] // 2

    sliceSlider1 = IntSlider(
        min=0, max=bmodeArray.shape[0], step=1, value=init_x,
        description='slice 1:',
        continuous_update=False,
        style={'description_width': '130px'}
    )
    sliceSlider2 = IntSlider(
        min=0, max=bmodeArray.shape[1], step=1, value=init_y,
        description='slice 2:',
        continuous_update=False,
        style={'description_width': '130px'}
    )
    sliceSlider3 = IntSlider(
        min=0, max=bmodeArray.shape[2], step=1, value=init_z,
        description='slice 3:',
        continuous_update=False,
        style={'description_width': '130px'}
    )

    sliceSlider1.axis = 'x'
    sliceSlider2.axis = 'y' 
    sliceSlider3.axis = 'z'

    def sliceChange(change):
        print('test')
        if change['owner'].axis == 'x':
            print('x')
            x_index = change['new']
            origin_x = origin[0] + x_index * spacing[0]
            new_slice = grid.slice(normal='x', origin=(origin_x, 0, 0))
            slice1.mapper.dataset.deep_copy(new_slice)
        elif change['owner'].axis == 'y':
            print('y')
            y_index = change['new']
            origin_y = origin[1] + y_index * spacing[1]
            new_slice = grid.slice(normal='y', origin=(0, origin_y, 0))
            slice2.mapper.dataset.deep_copy(new_slice)
        else:
            print('z')
            z_index = change['new']
            origin_z = origin[2] + z_index * spacing[2]
            new_slice = grid.slice(normal='z', origin=(0, 0, origin_z))
            slice3.mapper.dataset.deep_copy(new_slice)
        p.render()
    
    def toggleVis(change):
        change['owner'].actor.SetVisibility(not change['new'])
        p.render()

    p = pv.Plotter()
    origin_x = origin[0] + init_x * spacing[0]
    origin_y = origin[1] + init_y * spacing[1]
    origin_z = origin[2] + init_z * spacing[2]
    sl1 = grid.slice(normal='x', origin=(origin_x, 0, 0))
    sl2 = grid.slice(normal='y', origin=(0, origin_y, 0))
    sl3 = grid.slice(normal='z', origin=(0, 0, origin_z))
    slice1 = p.add_mesh(sl1, cmap='gray', clim=[0, 1])
    slice2 = p.add_mesh(sl2, cmap='gray', clim=[0, 1])
    slice3 = p.add_mesh(sl3, cmap='gray', clim=[0, 1])
    volume = p.add_volume(grid, cmap='gray_r', clim=[0, 1])
    p.show(jupyter_backend='trame')

    sliceSlider1.observe(sliceChange, names='value')
    sliceSlider2.observe(sliceChange, names='value')
    sliceSlider3.observe(sliceChange, names='value')
        
    hideVol = Checkbox(
    value=False,
    description='hide volume',
    disabled=False,
    indent=False, 
    )

    hideS1 = Checkbox(
    value=False,
    description='hide slice 1',
    disabled=False,
    indent=False
    )

    hideS2 = Checkbox(
    value=False,
    description='hide slice 2',
    disabled=False,
    indent=False
    )

    hideS3 = Checkbox(
    value=False,
    description='hide slice 3',
    disabled=False,
    indent=False
    )

    hideVol.actor = volume
    hideS1.actor = slice1
    hideS2.actor = slice2
    hideS3.actor = slice3

    hideVol.observe(toggleVis, names='value')
    hideS1.observe(toggleVis, names='value')
    hideS2.observe(toggleVis, names='value')
    hideS3.observe(toggleVis, names='value')

    display(VBox([sliceSlider1, sliceSlider2, sliceSlider3, HBox([hideVol, hideS1, hideS2, hideS3])]))
