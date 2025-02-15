# coding=utf-8
# Copyright 2022 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Caltech images dataset."""

import os

import numpy as np
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_LABELS_FNAME = "image_classification/caltech101_labels.txt"
# Original url should be
# http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
# which redirect to drive. We could use the original URL once
# `downloader.download` correctly handle drive URLs hidden behind a redirection.
_URL = "https://drive.google.com/uc?export=download&id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp"
_TRAIN_POINTS_PER_CLASS = 30


class Builder(tfds.core.GeneratorBasedBuilder):
  """Caltech-101."""

  VERSION = tfds.core.Version("3.0.1")
  RELEASE_NOTES = {
      "3.0.0": "New split API (https://tensorflow.org/datasets/splits)",
      "3.0.1": "Website URL update",
  }

  def _info(self):
    names_file = tfds.core.tfds_path(_LABELS_FNAME)
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(),
            "label": tfds.features.ClassLabel(names_file=names_file),
            "image/file_name": tfds.features.Text(),  # E.g. 'image_0001.jpg'.
        }),
        supervised_keys=("image", "label"),
        homepage="http://www.vision.caltech.edu/Image_Datasets/Caltech101/",
    )

  def _split_generators(self, dl_manager):
    path = dl_manager.download_and_extract(_URL)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "images_dir_path": path,
                "is_train_split": True,
            }),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                "images_dir_path": path,
                "is_train_split": False,
            }),
    ]

  def _generate_examples(self, images_dir_path, is_train_split):
    """Generates images and labels given the image directory path.

    As is usual for this dataset, 30 random examples from each class are added
    to the train split, and the remainder are added to the test split.

    Args:
      images_dir_path: path to the directory where the images are stored.
      is_train_split: bool, if true, generates the train split, else generates
        the test split.

    Yields:
      The image path, and its corresponding label and filename.

    Raises:
      ValueError: If too few points are present to create the train set for any
        class.
    """
    # Sets random seed so the random partitioning of files is the same when
    # called for the train and test splits.
    numpy_original_state = np.random.get_state()
    np.random.seed(1234)

    parent_dir = tf.io.gfile.listdir(images_dir_path)[0]
    walk_dir = os.path.join(images_dir_path, parent_dir)
    dirs = tf.io.gfile.listdir(walk_dir)

    for d in dirs:
      # Each directory contains all the images from a single class.
      if tf.io.gfile.isdir(os.path.join(walk_dir, d)):
        for full_path, _, fnames in tf.io.gfile.walk(os.path.join(walk_dir, d)):

          # _TRAIN_POINTS_PER_CLASS datapoints are sampled for the train split,
          # the others constitute the test split.
          if _TRAIN_POINTS_PER_CLASS > len(fnames):
            raise ValueError("Fewer than {} ({}) points in class {}".format(
                _TRAIN_POINTS_PER_CLASS, len(fnames), d))
          train_fnames = np.random.choice(
              fnames, _TRAIN_POINTS_PER_CLASS, replace=False)
          test_fnames = set(fnames).difference(train_fnames)
          fnames_to_emit = train_fnames if is_train_split else test_fnames

          for image_file in fnames_to_emit:
            if image_file.endswith(".jpg"):
              image_path = os.path.join(full_path, image_file)
              record = {
                  "image": image_path,
                  "label": d.lower(),
                  "image/file_name": image_file,
              }
              yield "%s/%s" % (d, image_file), record
    # Resets the seeds to their previous states.
    np.random.set_state(numpy_original_state)
