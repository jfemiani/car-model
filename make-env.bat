conda create -y -n car-model python pip affine docopt jupyter scipy gdal geos numpy mingw pillow scikit-image scikit-learn 
call activate car-model
conda install shapely cartopy -c IOOS
pip install progressbar