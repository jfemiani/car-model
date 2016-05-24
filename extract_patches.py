""" Extract Patches

This scripts reads in a shapefile and a raster image extracts patches to be used for machine learning algorithms.
If the shapefile consists of points, then a rectangular patch centered at each point is extracted. If the shapefile
consists of lines, then the first and second point are used to determine an orientation, and a patch is extracted
centered at the midpoint of the line and rotated so that the line would be horizontal, pointing in the +x direction.

The ouput folder will always contain a CSV file with the shapefile and raster name, as well as index of the feature in
the shapefile, the latitude and longitude, and image row and column of each feature.


Usage:
    extract_patches.py (-i <shapefile>)...  -r <raster> [--odir <outdir>] [options]
    extract_patches.py -h

Options:
    -i <shapefile>           A shapefile with features indicated. Wildcards are allowed.
    -r <raster>              A  raster source image to extract features from. No wildcards -- if you have multiple
                             raster files consider using `gdalbuildvrt` first to create a virtual mosaic.
    --odir <outdir>          The output folder. [default: ./patches/]
    --otif                   Outout TIF files [default]
    --ojpg                   Output JPG files
    --size <width>,<height>  The size of each extracted patch, in pixels. Please make sure there are no space characters
                             between the arguments unless you put it in quotes! [default: 64,64]
    --scale <scale>          The amount to scale each image by before extracting the patches (e.g.  if the raster is
                             in a higher resolution you may choose to scale it down). [default: 1.0]
    --csv <csvfile>          Create a CSV file with the center lat/lon, center pixel xy, angle, and the patch filename
                             as well as the name of the shapefile source and the raster spource image.
    --noprogress             Suppress progress updates.
    --vclip <min,max>        Clip input values to `min` and `max`. Mainly useful when floating point output is not an
                             option (e.g. JPEG output).
    --vstretch <min,max>     Stretch output values to 'min' and 'max', after clipping. Mainly useful when floating point
                             output is not possible. Note that JPEG output must be in 0 to 1.
    --debug                  Set the log level to DEBUG
    --info                   Set the log level to INFO
    --logfile <filename>     Log to a file [default: '']

"""
import hashlib
import os

import numpy
import osgeo
import rasterio
from affine import Affine
from math import degrees, atan2, hypot, floor, ceil
from osgeo import ogr, gdalnumeric
from osgeo.osr import SpatialReference
from rasterio._io import RasterReader, RasterUpdater
# from skimage.transform import rescale
from scipy.ndimage import rotate, zoom
from skimage.util.dtype import img_as_ubyte

BREAKING_RELEASE = 0  # Big, rare, when you actually remove deprecated code.
NONBREAKING_RELEASE = 0  # Upload new binaries when you increment this, much less frequently then minor
MINOR_UPDATE = 1  # Increment for pullers
EXPERIMENTAL = 1

VERSION = '{}.{}.{}'.format(BREAKING_RELEASE, NONBREAKING_RELEASE, MINOR_UPDATE)
if EXPERIMENTAL:
    VERSION += 'a{}'.format(EXPERIMENTAL)
__version__ = VERSION

from glob import glob
from docopt import docopt
import logging
from progressbar import Percentage, Bar, ETA, ProgressBar


def collect_filenames(wildcard_list):
    logger = logging.getLogger(__name__)
    results = []
    for wildcard in wildcard_list:
        matching_files = glob(wildcard)
        if len(matching_files) is 0:
            logger.warning("No files matching input specification '{}'".format(wildcard))
        for filename in matching_files:
            results.append(filename)
    return results


def main():
    args = docopt(__doc__, version=VERSION)

    logparams = {}
    if args['--debug']:
        logparams.update(level=logging.DEBUG)
    elif args['--info']:
        logparams.update(level=logging.INFO)
    else:
        logparams.update(level=logging.CRITICAL)

    if args['--logfile'] != '':
        logparams.update(filename=args['--logfile'])

    logging.basicConfig(**logparams)

    logger = logging.getLogger('extract_patches')
    logger.debug('input \n {}'.format(args))

    assert isinstance(logger, logging.Logger)

    shapefiles = collect_filenames(args['-i'])
    if len(shapefiles) == 0:
        logger.error('No matching shapefiles for inoput `{}`'.format(args['-i']))
        return

    raster = args['-r']

    try:
        size = [int(x) for x in args['--size'].split(',')]
        patch_width, patch_height = size
        logger.debug("Set patch size to {} x {}".format(patch_width, patch_height))
    except:
        logger.error("Unable to parse option '--size'")
        return

    try:
        scale = float(args['--scale'])
        assert scale > 0
        logger.debug("Set scale to {}".format(scale))
    except:
        logger.error("Unable to parse option '--scale'")
        return

    silent = args['--noprogress']

    output_folder = args['--odir']
    try:
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
            logger.debug("Created output folder '{}'".format(output_folder))
        else:
            logger.debug("Found existing output folder '{}'".format(output_folder))
    except:
        logger.error("Unable to find or create output directory `{}`".format(output_folder))
        return

    if args['--ojpg']:
        fmt = '.jpg'
    else:  # args['--otif']  (default)
        fmt = '.tif'
    logger.debug("Output format set to {}".format(fmt))

    clip = args['--vclip'] is not None
    if clip:
        clipmin, clipmax = [float(x) for x in args['--vclip'].split(',')]
        logger.debug("Clipping output to [{}, {}]".format(clipmin, clipmax))
    else:
        clipmin, clipmax = 0, 1
        logger.debug("Not clipping output -- assuming range of value is [{},{}]".format(clipmin, clipmax))

    stretch = args['--vstretch'] is not None
    if stretch:
        stretchmin, stretchmax = [float(x) for x in args['--vstretch'].split(',')]
        logger.debug("Output value range will be stretched to [{},{}]".format(stretchmin, stretchmax))
    else:
        logger.debug("Output values will not be stretched")

    if args['--csv']:
        csv_file_name = args['--csv']
        if os.path.isfile(csv_file_name):
            logger.error("CSV File already exists; please remove or rename it first.")
            logger.debug("Writing to CSV File '{}'".format(csv_file_name))
            return
    else:
        csv_file_name = None
        logger.debug("No CSV output")

    # Estimate number of shape features
    count = 0
    if not silent:
        pbar = ProgressBar(len(shapefiles), ['Counting Features:', Percentage(), ' ', Bar(), ' ', ETA()])
        pbar.start()
    for i, s in enumerate(shapefiles):
        vector = ogr.Open(s)
        layer = vector.GetLayer()
        count += layer.GetFeatureCount()
        if not silent:
            pbar.update(i)
    if not silent:
        pbar.finish()

    logger.debug("Counted {} features in {} shapefiles".format(count, len(shapefiles)))

    # Write header for CSV file
    if csv_file_name is not None:
        with open(os.path.join(output_folder, csv_file_name), 'w') as csvf:
            csvf.write('gx, gy, r1, r2, theta, patch_width, patch_height, image_namei\n')

    with rasterio.open(raster) as rf:
        assert isinstance(rf, RasterReader)
        srs = SpatialReference(str(rf.crs_wkt))
        affine = rf.affine
        geo_to_pixels = ~affine

        logging.debug("Output CRS will be '''{}'''".format(srs.ExportToPrettyWkt()))

        if not silent:
            pbar = ProgressBar(count, ['Exporting Patches:', Percentage(), ' ', Bar(), ' ', ETA()])
            pbar.start()
        for sf in shapefiles:
            logger.info("Processing input '{}'".format(sf))
            vector = ogr.Open(sf)
            assert isinstance(vector, ogr.DataSource)

            layer = vector.GetLayer()
            assert isinstance(layer, ogr.Layer)

            if not srs.IsSame(layer.GetSpatialRef()):
                logger.warning("Coordinate system mismatch (its ok, I will reproject)")

            for f in layer:
                if not silent:
                    pbar.update(pbar.currval + 1)
                geom = f.GetGeometryRef()
                assert isinstance(geom, ogr.Geometry)
                geom = geom.TransformTo(srs)
                points = geom.GetPoints()
                source = points[0]
                target = points[-1]
                sx, sy = geo_to_pixels * source
                tx, ty = geo_to_pixels * target
                if len(points) == 2:
                    cx, cy = (sx + tx) / 2, (sy + ty) / 2
                else:
                    cx, cy = geo_to_pixels * points[1]
                dx, dy = (tx - sx), (ty - sy)
                theta = degrees(atan2(dy, dx))  # In PIXELS, CCW from +x. Not necessarily CCW from E (or CW from N)
                r1 = hypot(tx - cx, ty - cy)
                r2 = hypot(cx - sx, cy - sy)
                r1, r2 = max(r1, r2), min(r1, r2)  # For 3 points, we assume two radii. Else these are duplicates.
                gx, gy = affine * (cx, cy)  # Geographic coordinates (e.g. lat lon) of the center.

                # We read a square slightly larger than the scaled version of our patch, so that
                # we can safely rotate the raster without missing pixels in the corners.

                box_radius = hypot(patch_width, patch_height) / (2.0 * scale)
                x0, x1 = int(floor(cx - box_radius)), int(ceil(cx + box_radius))
                y0, y1 = int(floor(cy - box_radius)), int(ceil(cy + box_radius))

                # save patch...

                kwargs = rf.meta
                patch_affine = (affine * Affine.translation(cx, cy) *
                                Affine.rotation(angle=-theta) * Affine.translation(-patch_width / 2.,
                                                                                   -patch_height / 2.))

                if fmt == '.tif':
                    kwargs.update(
                        driver='GTiff',
                        compress='lzw',
                        dtype=numpy.float32
                    )
                elif fmt == '.jpg':
                    kwargs.update(
                        driver='JPEG',
                        quality=90,
                        dtype=numpy.uint8
                    )

                kwargs.update(
                    transform=patch_affine,
                    width=patch_width,
                    height=patch_height
                )

                box_radius *= scale
                name = hashlib.md5(str(patch_affine) + raster).hexdigest()
                image_name = os.path.join(output_folder, name + fmt)

                if csv_file_name is not None:
                    with open(os.path.join(output_folder, csv_file_name), 'a+') as csvf:
                        fields = gx, gy, r1, r2, theta, patch_width, patch_height, image_name
                        csvf.write(','.join([str(_) for _ in fields]) + '\n')

                with rasterio.open(image_name, 'w', **kwargs) as pf:
                    assert isinstance(pf, RasterUpdater)
                    for band in range(rf.count):
                        patch = rf.read(band + 1, window=((y0, y1), (x0, x1)), boundless=True, )
                        patch = patch.astype(numpy.float32)
                        patch_rotated = rotate(patch, theta, reshape=False)
                        patch_scaled = zoom(patch_rotated, scale)
                        i0 = int(round(box_radius - patch_height / 2.))
                        i1 = i0 + patch_height
                        j0 = int(round(box_radius - patch_width / 2.))
                        j1 = j0 + patch_width
                        patch_cropped = patch_scaled[i0:i1, j0:j1]

                        if clip:
                            patch_cropped = numpy.clip(patch_cropped, clipmin, clipmax)
                        if stretch:
                            patch_cropped = (patch_cropped - clipmin) / (clipmax - clipmin)
                            patch_cropped = patch_cropped * (stretchmax - stretchmin) + stretchmin

                        if fmt == '.jpg':
                            # JPEG does not support floating point output. All we can do is 8 bit
                            # (python has not 12bit array type)
                            patch_cropped = img_as_ubyte(patch_cropped.clip(-1, 1))
                        pf.write(patch_cropped, band + 1)
        if not silent:
            pbar.finish()

        logger.debug("Finished.")

if __name__ == '__main__':
    main()
