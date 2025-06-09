from ipywidgets import IntSlider, FloatSlider, interactive_output, Output, Layout, HBox, VBox, Dropdown, Button, ToggleButtons
from IPython.display import display
import matplotlib as plt

from preprocessing import standardSVD, randomSVD, adaptiveSVD, reconstructSVD
from visualization import makeBmode, makeFlow, plotBlood, plotPowerDoppler

bloodOut = Output()
powerDopplerOut = Output()
standardSVDOutput = Output()

def power2D(rawImages):
    shape = rawImages.shape
    U, S, Vh = standardSVD(rawImages)
    def update(tissue_threshold, noise_threshold, maxMag, minMag, sigma, svdMode):
        if svdMode == 'Standard':
            standInputs.layout.display = 'flex'
            randInputs.layout.display = 'none'
            adaptiveInputs.layout.display = 'none'
        elif svdMode == 'Random':
            standInputs.layout.display = 'none'
            randInputs.layout.display = 'flex'
            adaptiveInputs.layout.display = 'none'
        else:
            standInputs.layout.display = 'none'
            randInputs.layout.display = 'none'
            adaptiveInputs.layout.display = 'flex'
        with bloodOut:
            bloodOut.clear_output(wait=True)
            plt.figure(figsize=(5, 5))
            filteredImages = reconstructSVD(U, S, Vh, tissue_threshold, noise_threshold, shape)
            plt.title('Filtered Blood Signal')
            plt.imshow(makeFlow(filteredImages[0:16], maxMag, 0, sigma))
            plt.show()
        with powerDopplerOut:
            powerDopplerOut.clear_output(wait=True)
            bmodeArray = makeBmode(rawImages)
            plt.figure(figsize=(5, 5))
            plt.title('Power Doppler')
            plt.imshow(bmodeArray, cmap='grey')
            rgbaFlow = makeFlow(filteredImages[0:16], maxMag, minMag, sigma)
            plt.imshow(rgbaFlow)
            plt.show()

    tissueSlider = IntSlider(
        min=0, max=256, step=1, value=0,
        description='Tissue Threshold:',
        continuous_update=False,
        style={'description_width': '130px'},
        layout=Layout(width='400px')
    )

    noiseSlider = IntSlider(
        min=0, max=256, step=1, value=256,
        description='Noise Threshold:',
        continuous_update=False,
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
        style={'description_width': '130px'},
        layout=Layout(width='400px', padding='50px 0px 0px 0px')
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
    )

    ensembleSize = IntSlider(
        min=0, max=256, step=1, value=256,
        description='Ensemble Size:',
        continuous_update=False,
        style={'description_width': '130px'}
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


    interactive_output(update, {'tissue_threshold': tissueSlider, 'noise_threshold': noiseSlider, 'maxMag': maxPowerSlider, 'minMag': minPowerSlider, 'sigma':sigmaSlider, 'svdMode':svdSelector})

    standInputs = VBox([standardSVDOutput, tissueSlider, noiseSlider, sigmaSlider, maxPowerSlider])

    randInputs = VBox([randomK, randomIters, randomD])

    adaptiveInputs = VBox([adaptiveBlockSize, adaptiveBlockOverlap, adaptiveTissueThreshold, adaptiveNoiseThreshold])

    display(HBox([VBox([ensembleSize, svdSelector, standInputs, randInputs, adaptiveInputs, minPowerSlider]), bloodOut, powerDopplerOut]))

def power3D():
    return