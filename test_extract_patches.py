import hashlib
import os
import shutil
import unittest
from urllib import urlretrieve
from zipfile import ZipFile

import subprocess

def md5(fname):
    return hashlib.md5(open(fname, 'rb').read()).hexdigest()

class TestExtractPatches(unittest.TestCase):
    def setUp(self):
        # Fetch test data
        if not os.path.isfile('data/cars_set_1.zip'):
            os.makedirs('data')
            urlretrieve('http://users.miamioh.edu/femianjc/cars_set_1.zip', 'data/cars_set_1.zip')
        if not os.path.isdir('data/cars_set_1'):
            ZipFile('cars_set_1.zip').extractall('data/cars_set_1')


    def test_cars(self):
        """
        Extracts all of the cars as JPG images, stretching the colors so that they can be displayed properly.
        Outputs a CSV file with information on each patch.

        Then is compares the MD5 checksums for a few randomly generated files against reference checksums to make
        sure this output is identical to a valid output.

        :return:
        """
        if os.path.exists('data/cars_set_1/output'):
            shutil.rmtree('data/cars_set_1/output')

        subprocess.check_call(('python', 'extract_patches.py',
                               '-i', 'data/cars_set_1/1/cars.shp',
                               '-r', 'data/cars_set_1/1.tif',
                               '--size', '80,40',
                               '--odir', 'data/cars_set_1/output',
                               '--ojpg',
                               '--vclip', '0,255',
                               '--vstretch', '0,1',
                               '--csv', 'table.csv'))


        # Now compare some output files to what we expect

        # Rather than compare images I just keep track of the md5 checksums for a few examples
        # These may need to be updates if we change the way images are produced or named
        expected_checksum = '60db30fb112126799f0e4b6492946d0f'
        self.assertEqual(expected_checksum, md5('data/cars_set_1/output/497974_006777E-5457505_30236N-80x40.jpg'))

        expected_checksum = 'a5b5ee2c7c18f3e5b9225badab9df4e4'
        self.assertEqual(expected_checksum, md5('data/cars_set_1/output/498089_252034E-5457540_74051N-80x40.jpg'))

        expected_checksum = '49768f2caff38640fcf27e06b677adc3'
        self.assertEqual(expected_checksum, md5('data/cars_set_1/output/498042_832905E-5457422_55389N-80x40.jpg'))

