# car-model



## Setup From Source

To create a new anaconda environment with all our dependancies...

On _Windows_, try
```
conda create -y -n car-model python pip affine rasterio docopt jupyter scipy geos numpy mingw pillow scikit-image scikit-learn 
call activate car-model
conda install shapely cartopy gdal -c IOOS
pip install progressbar
```

For _Linux_, the same _may_ also work, if you replace 'call' by 'source'
```bash
source activate car-model
```

