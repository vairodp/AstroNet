"""ska dataset."""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from astropy.io import fits
import tensorflow_datasets as tfds

from data_prep import divide_images
from configs.train_config import IMG_SIZE

PATHS = {
  'B1_1000h': '../data/raw/SKAMid_B1_1000h_v3.fits',
  'B2_1000h': '../data/raw/SKAMid_B2_1000h_v3.fits',
  'ANNOT_B1': '../data/raw/TrainingSet_B1_v2_ML.txt',
  'ANNOT_B2': '../data/raw/TrainingSet_B2_v2_ML.txt',
  'CUTOUTS_B1': '../data/training/B1_1000h/',
  'CUTOUTS_B2': '../data/training/B2_1000h/',
  'TRAINING': '../data/training'
}

_DATA_B1_1000h = 'https://owncloud.ia2.inaf.it/index.php/s/hbasFhd4YILNkCr/download'
_ANNOTATIONS_B1_1000h = 'https://owncloud.ia2.inaf.it/index.php/s/I4HL84Etgw9c2Le/download'

_DATA_B2_1000h = 'https://owncloud.ia2.inaf.it/index.php/s/GsxoTyv1zrdRTu4/download'
_ANNOTATIONS_B2_1000h = 'https://owncloud.ia2.inaf.it/index.php/s/I4HL84Etgw9c2Le/download'


_DESCRIPTION = """
The SKA Science Data Challenge #1 (SDC1) release consists of 9 files, 
with the format of FITS images. Each file is a simulated SKA continuum image 
in total intensity of the same field at 3 frequencies (560 MHz, representative 
of SKA Mid Band 1, 1.4 GHz, representative of SKA Mid Band 2 and 9.2 GHz, 
representative of SKA Mid Band 5) and 3 telescope integrations (8, 100, 1000 h 
as representative of a single, medium-depth and deep integration, respectively).
"""

_CITATION = """
@misc{bonaldi2018square,
      title={Square Kilometre Array Science Data Challenge 1},
      author={Anna Bonaldi and Robert Braun},
      year={2018},
      eprint={1811.10454},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM}
}
"""


class SKA(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for ska_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # Features of our dataset: images, labels ...
            'image': tfds.features.Tensor(shape=(IMG_SIZE,IMG_SIZE), dtype=tf.float64),
            'image/filename' : tfds.features.Text(),
            'objects': tfds.features.Sequence({
                'bbox': tfds.features.BBoxFeature(),
                'label': tfds.features.ClassLabel(names=['SS-AGN', 'FS-AGN','SFG'])
            })        
        }),
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://astronomers.skatelescope.org/ska-science-data-challenge-1/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # Data has already been downloaded
    #paths = dl_manager.download([_DATA_B1_1000h, 
                                #_DATA_B2_1000h, 
                                #_DATA_B5_1000h,
                                #_ANNOTATIONS_B1_1000h,
                                #_ANNOTATIONS_B2_1000h,
                                #_ANNOTATIONS_B5_1000h])
    #self._rename_paths(paths)

    divide_images(PATHS['B1_1000h'], PATHS['ANNOT_B1'], PATHS['CUTOUTS_B1'])
    divide_images(PATHS['B2_1000h'], PATHS['ANNOT_B2'], PATHS['CUTOUTS_B2'])
  

    return {
        'train': self._generate_examples(tfds.core.ReadWritePath(PATHS['TRAINING'])),
    }
  
  def _rename_paths(self, paths):
    for path in paths:
      dir_path = "../data/raw/"
      info_path = dir_path + path.name + '.INFO'
      with open(info_path, mode='r') as info_file:
        info = json.load(info_file)
      path.rename(dir_path + info["original_fname"])

  def _generate_examples(self, path):
    """Yields examples."""
    folders = ['B1_1000h', 'B2_1000h']
    for folder in folders:
      folder_path = path / folder
      for img_path in [img for img in os.listdir(folder_path._path_str) if img.endswith('.fits')]:
        fits_data = fits.open(folder_path._path_str + '/' + img_path)
        fits_data = fits_data[0].data.astype(np.float64)
        yield img_path, {
            'image': fits_data,
            'image/filename': img_path,
            'objects': self._generate_galaxies(folder_path._path_str, img_path)
          }

  def _generate_galaxies(self, folder_path, img_path):
    csv_path = folder_path + '/galaxies.csv'
    galaxies_df = pd.read_csv(csv_path)
    objs = galaxies_df[galaxies_df.img_path == img_path]
    galaxies = []
    for _, obj in objs.iterrows():
      galaxies.append({
        'bbox': tfds.features.BBox(
          ymin = obj['ymin'],
          xmin = obj['xmin'],
          ymax = obj['ymax'],
          xmax = obj['xmax']
        ),
        'label': obj['class']
      })
    return galaxies