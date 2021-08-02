"""ska dataset."""

import os
import json
import pandas as pd
import tensorflow_datasets as tfds

from configs.yolo_v4 import IMG_SIZE
from data_prep import newDivideImages

PATHS = {
  'B1_1000h': '../data/raw/SKAMid_B1_1000h_v3.fits',
  'B2_1000h': '../data/raw/SKAMid_B2_1000h_v3.fits',
  'B5_1000h': '../data/raw/SKAMid_B5_1000h_v3.fits',
  'ANNOT_B1': '../data/raw/TrainingSet_B1_v2.txt',
  'ANNOT_B2': '../data/raw/TrainingSet_B2_v2.txt',
  'ANNOT_B5': '../data/raw/TrainingSet_B5_v2.txt',
  'CUTOUTS_B1': '../data/training/B1_1000h/',
  'CUTOUTS_B2': '../data/training/B2_1000h/',
  'CUTOUTS_B5': '../data/training/B5_1000h/',
  'TRAINING': '../data/training'
}

_DATA_B1_1000h = 'https://owncloud.ia2.inaf.it/index.php/s/hbasFhd4YILNkCr/download'
_ANNOTATIONS_B1_1000h = 'https://owncloud.ia2.inaf.it/index.php/s/iTOVkIL6EfXkcdR/download'

_DATA_B2_1000h = 'https://owncloud.ia2.inaf.it/index.php/s/GsxoTyv1zrdRTu4/download'
_ANNOTATIONS_B2_1000h = 'https://owncloud.ia2.inaf.it/index.php/s/0HMJmNhPywxQdY4/download'

_DATA_B5_1000h ='https://owncloud.ia2.inaf.it/index.php/s/nK8Pqf3XIaXFuKD/download'
_ANNOTATIONS_B5_1000h = 'https://owncloud.ia2.inaf.it/index.php/s/Y5CIa5V3QiBu1M1/download'

# TODO(ska_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
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
    # TODO(ska_dataset): add galaxy properties at some point
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(IMG_SIZE, IMG_SIZE, 3)),
            'image/filename' : tfds.features.Text(),
            'objects': tfds.features.Sequence({
                'bbox': tfds.features.BBoxFeature(),
                'label': tfds.features.ClassLabel(names=['SS-AGN', 'FS-AGN','SFG'])
            })        
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
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

    #print(paths)

    #self._rename_paths(paths)
    #print(paths)
    newDivideImages(PATHS['B1_1000h'], PATHS['ANNOT_B1'], PATHS['CUTOUTS_B1'])
    newDivideImages(PATHS['B2_1000h'], PATHS['ANNOT_B2'], PATHS['CUTOUTS_B2'])
    #newDivideImages(PATHS['B5_1000h'], PATHS['ANNOT_B5'], PATHS['CUTOUTS_B5'])

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
    folders = ['B1_1000h', 'B2_1000h', 'B5_1000h']
    for folder in folders:
      folder_path = path / folder
      print(folder_path._path_str)
      for img_path in folder_path.glob('*.png'):
        yield img_path.name, {
            'image': img_path,
            'image/filename': img_path.name,
            'objects': self._generate_galaxies(folder_path._path_str, img_path.name)
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
