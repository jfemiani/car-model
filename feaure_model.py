""" Transform images to feature vectors.

This script processes a list of images and produces a table of feature vectors based on them. The feature vectors
are all the same length, and they are formatted so that they can be loaded into tools like Orange.

USAGE:
    feature_model.py  pca transform -i <images>  --model <model> [options]
    feature_model.py  pca update -i <images>  --model <model>  --count  <count> [options]
    feature_model.py  pca build -i <images>  --model <model>  --count  <count> [options]
    feature_model.py  spca transform -i <images> --model <model> [options]
    feature_model.py  spca build -i <images> --model <model> [options]

OPTIONS:
    -i, --input <images>        Specify a list of raster files to generate features from. [default: *.tif]
    --model  <model>            A saved model. These are created when you pass `build` and they are used to determine
                                feature values when you pass `transform` as an argument.

    --count  <count>            The desired number of features (e.g the number of principle components in PCA). If this is
                                not set then it will be estimated.
                                [default:None]
    --debug                     Set the log level to debug.
    --info                      Set the log level to info.
    --logfile <filename>        Log to a file. [default: None]

    --jobs <njobs>              The number of concurrent processes to spawn (e.g. number of CPUS)  [default: 1]
"""

# --rotate <range>            Augment data by rotating input around the center using random rotations. The range
#                             is specified as either  U,count,min,max  or  N,count,mean,std. No spaces.
# --translate-x <range>       Augment data by translating input using random offsets. The range
#                             is specified as either  U,count,min,max  or  N,count,mean,std. No spaces.
# --translate-y <range>       Augment data by translating input using random offsets. See the documentation for
#                             translate-x.
# --mirror-y                  Augment data by flipping vertically
# --mirror-x                  Augment data by flipping horizontally
# --whitenoise <range>        Augment data by adding white noise. The range of noise added to each band of each pixel
#                             is specified as either  U,count,min,max  or  N,count,mean,std. No spaces.
import logging
from glob import glob
import rasterio
from docopt import docopt
from sklearn.decomposition import SparsePCA, PCA
import numpy as np
import progressbar
import pickle

def main():
    args = docopt(__doc__)

    log_params = {}
    if args['--debug']:
        log_params.update(level=logging.DEBUG)
    elif args['--info']:
        log_params.update(level=logging.INFO)
    else:
        log_params.update(level=logging.CRITICAL)

    if args['--logfile'] != '':
        log_params.update(filename=args['--logfile'])

    logging.basicConfig(**log_params)
    logger = logging.getLogger('extract_patches')
    logger.debug('input \n {}'.format(args))

    images = glob(args['-i'])
    if len(images) == 0:
        logger.error("No input images specified")
        return

    jobs = int[args['--jobs']]
    components = int[args.get('--count', 0)]

    model_path = args['--model']


    if args['spca']:
        model_class = SparsePCA
    else:  # args['pca']:
        model_class = PCA

    with rasterio.open(images[0]) as rf:
        width = rf.width
        height = rf.height
        bands = rf.bands

    logging.debug("Raster shape is {}x{}x{}".format(bands, width, height))
    logging.debug("Reading {} raster files".format(len(images)))
    pbar = progressbar.ProgressBar(len(images), ['Reading Rasters:', progressbar.Percentage(), ' ',
                                                 progressbar.Bar(), ' ', progressbar.ETA()])
    pbar.start()
    data = np.empty((len(images), width*height*bands))
    for i, img in enumerate(images):
        with rasterio.open(img) as rf:
            car = rf.read()
            data[i] = car.flatten()
        pbar.update(i)
    pbar.finish()

    xdata = None
    if args['build']:
        logging.info("Building model")
        pbar = progressbar.ProgressBar(len(images), ['Building Model:', progressbar.Percentage(), ' ',
                                                     progressbar.Bar(), ' ', progressbar.ETA()])

        model = model_class(n_components=components, n_jobs=jobs, verbose=True)
        xdata = model.fit_transform(data)

        with open(model_path, 'wb') as mf:
            pickle.dump(model, mf)
    elif args['transform']:
        logging.info("Transforming")
        with open(model_path, 'rb') as mf:
            model = pickle.load(mf)
        xdata = model.transform(data)

    if xdata:
        np.savetxt('xdata.csv', xdata, delimiter=',')