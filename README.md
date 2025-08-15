# Zemp Lab Power Doppler

This repository filters tissue and noise from ultrasound beamformed pulse-wave signals and renders 2D and 3D Power Doppler images.

To use this repository, ensure you have python and pip installed on your computer. 

Run this using command promt to create a virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Filtering Data

**Filtering Methods** can be called on the "run.py" file, which contains functions that call filtering functions from "preprocessing.py", plots power doppler if data is 3D, i.e. (x, z, ensemble), and saves the filtered data. 
NOTE: if data is 3D, i.e. (x, z, y, ensemble), the 3D Power Doppler visualizer in visualizer.ipynb must be used for interactive 3D visualization. 

**UI functionality for 2D PD:**

A basic UI for filtering and visualizing 2 dimensional power doppler can be found in visualizer.ipynb. 

It computes a standard SVD on the matrix and generates a **preview of the power Doppler image** along with a B-mode image. Once the “Filter” button is pressed, the SVD range and smoothing parameter (σ) can be adjusted, and the power Doppler preview updates in real time. Additionally, the minimum power threshold can be lowered to overlay the power Doppler image onto the B-mode image.

**Note:** This viewer is optimized only for previewing the SVD basis and getting a quick sense of the blood signal. For more complex processing, filtering, or analysis, it is recommended to use the command-line interface or create custom scripts using the filtering functions in "preprocessing.py". 

To use, insert the file path and matrix variable and run the code window in the notebook by pressing the “play” icon. 

```python
import ui  
import numpy as np  
import importlib  
from scipy.io import loadmat  
import visualization

importlib.reload(ui)  
importlib.reload(visualization)

# insert file name and matrix variable  
data = loadmat('file.mat')['var']

ui.power2D(data)
```

The UI window will appear below. 

**UI functionality for 3D PD:**

A basic UI for visualizing 3 dimensional power doppler can be found in visualizer.ipynb. 

It renders a 3D power doppler image from a filtered 4D matrix both as a crossplane visualization and a volume. Slice positions can be adjusted using the sliders and they can be hidden using the check boxes. 

To use, insert the file path and matrix variable and run the code window in the notebook by pressing the “play” icon. 

```python
import ui  
import numpy as np  
import importlib  
from scipy.io import loadmat

importlib.reload(ui)

# insert file name and matrix variable  
data = loadmat('file.mat')['var']

# custom spacing between datapoints and origin for rendering  
spacing = (1, 1, 1)  
origin = (0, 0, 0)

ui.power3D(data, spacing, origin)
```

The UI window will appear below. 


### Notes / Caveats

- You are welcome to modify, add features, or fix things as needed.  

- This repository is useful if you want to explore Power Doppler or to reference if you want to develop your own implementation.  