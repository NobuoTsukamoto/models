# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
r"""Downloads and converts iNat data to TFRecords of TF-Example protos.

This module downloads the iNat data, uncompresses it, reads the files
that make up the iNat data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take several minutes to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import os
import contextlib2

import numpy as np
import PIL.Image

import tensorflow as tf

from datasets import dataset_utils

# The URL where the Flowers data can be downloaded.
_TRAIN_ASIA_DATA_URL = (
    "https://storage.googleapis.com/inat_data_2018_asia/train_val2018.tar.gz"
)
_TRAIN_ASIA_ANNOTATION_URL = (
    "https://storage.googleapis.com/inat_data_2018_asia/train2018.json.tar.gz"
)
_VAL_ASIA_ANNOTATION_URL = (
    "https://storage.googleapis.com/inat_data_2018_asia/val2018.json.tar.gz"
)
_UN_OBFUSCATED_NAME_URL = "http://www.vision.caltech.edu/~gvanhorn/datasets/inaturalist/fgvc5_competition/categories.json.tar.gz"

flags = tf.app.flags
tf.flags.DEFINE_string("temp_dir", "./temp/", "Temp data directory.")
tf.flags.DEFINE_string("output_dir", "./output/", "Output data directory.")
tf.flags.DEFINE_string("super_category", "Aves", "Super Category")

FLAGS = flags.FLAGS


def _clean_up_temporary_files(dataset_dir):
    """ Removes temporary files used to create the dataset.

    Args:
        dataset_dir: The directory where the temporry files are stored.
    """
    tar_file_path = os.path.join(dataset_dir, "*.tar.gz")
    tf.gfile.Remove(tar_file_path)

    json_file_path = os.path.join(dataset_dir, "*.json")
    tf.gfile.Remove(json_file_path)


def _open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """ Opens all TF-Record shards for writing and add then to an exit stack.

    Args:
        exit_stack: A context2.ExitStack used to automatically closed the TF-Records opened in this function.
        base_path: The base path for all shards.
        num_shards: The number of shards.
    """
    tf_record_output_file_names = [
        "{}-{:05d}-of-{:05d}".format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    tf_records = [
        exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
        for file_name in tf_record_output_file_names
    ]

    return tf_records


def _create_category_index(categories, target_category):
    """ Creates dictionary of COCO compatible categories keyed by category id.

    Args:
        categories: a list of dicts, each of which has the following keys:
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.
        target_category: Targe super category.
    Returns:
        category_index: a dict containing the same entries as categories, but keyed
        by the 'id' field of each category.
    """
    category_index = {}
    for cat in categories:
        if cat["supercategory"] == target_category:
            category_index[cat["id"]] = cat
    return category_index


def _create_label_file(annotaion_file, output_path, target_category):
    """ Creates label file form annotation_file.

    Args:
        annotation_file: JSON file.
        output_dir: path to output to label file.
        target_category: Taregt super category.
    """
    # tf.gfile.GFile(annotations_file, 'r') as fid:
    # category_data = json.load(fid)
    # category_names = []
    # for cat in category_data:
    #     if cate['supercategory'] == target_category:
    #        category_names.append('')
    # category_index = _create_category_index(category_data['categories'], target_category)


def create_tf_example(image, annotations, image_dir, category_index):
    """ Converts image and annotations to a tf.Example proto.
    
    Args:
        image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
            u'width', u'date_captured', u'flickr_url', u'id']
        annotations: list of dicts with keys: [u'id', u'image_id', u'category_id']
        image_dir: directory containing the image files.
        category_idex: a dict containing COCO cateogory information keyed
            by the 'id' field of each category.
            
    Returns:
        example: The converted tf.Example

    Raises:
        VauleError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    height = image["height"]
    width = image["width"]
    file_name = image["file_name"]
    category_id = annotations["category_id"]
    class_id = annotations[category_id]["label_index"]

    full_path = os.path.join(image_dir, file_name)
    image_data = tf.gfile.GFile(full_path, "rb").read()

    example = dataset_utils.image_to_tfexample(
        image_data, b"jpg", height, width, class_id
    )

    return example


def _create_tf_record_from_coco_annotations(
    annotations_file,
    category_index,
    image_dir,
    output_path,
    target_category,
    num_shards,
):
    """ Loads COCO annotaion json files and converts to TF-Record format.

    Args:
        annotations_file: JNSON file.
        category_index: Dictionary containing categories.
        image_dir: Directory containing the image files.
        output_path: Path to output TF-Record file.
        num_shards: Number of output file shards.
    """
    with contextlib2.ExitStack() as tf_reord_close_stack, tf.gfile.GFile(
        annotations_file, "r"
    ) as fid:
        output_tfrecords = _open_sharded_output_tfrecords(
            tf_reord_close_stack, output_path, num_shards
        )
        ground_truth_data = json.load(fid)

        # Extract annotations present in image file.
        images = ground_truth_data["images"]
        target_images = {}
        for image in images:
            file_name = image["file_name"]
            full_path = os.path.join(image_dir, file_name)
            if tf.gfile.Exists(full_path):
                target_images[image["id"]] = image

        # Get annotation index form 'annotation' and categorn index
        annotations_index = {}
        for annotation in ground_truth_data["annotations"]:
            if annotation["category_id"] in category_index:
                image_id = annotation["image_id"]
                if image_id in target_images:
                    annotations_index[image_id] = annotation

        for idx, image in enumerate(target_images):
            if idx % 100 == 0:
                tf.logging.info("On image %d of %d", idx, len(images))

            annotations_list = annotations_index[image["id"]]
            tf_example = create_tf_example(
                image, annotations_list, image_dir, category_index
            )
            shard_idx = idx % num_shards
            output_tfrecords[shard_idx].write(tf_example.SerializeToString())
        tf.logging.info("Finished writing")


def _create_label_from_coco_annotations(
    annotations_file, un_obfuscated_name_file, image_dir, target_category
):
    """ Create label dic from COCO annotaion json files.

    Args:
        annotations_file: JSON file.
        un_obfuscated_name_file: JSON file(Un-obfuscated names).
        image_dir: Directory containing the image files.
        target_category: Target category

    Returns:
        category index.
    """
    with tf.gfile.GFile(annotations_file, "r") as fid1, tf.gfile.GFile(
        un_obfuscated_name_file, "r"
    ) as fid2:
        ground_truth_data = json.load(fid1)
        un_obfuscated_name = json.load(fid2)

        # Get category index from 'categories'
        category_index = _create_category_index(
            un_obfuscated_name, target_category
        )

        print('Num of categoies : ', len(category_index))

        # Extract annotations present in image file.
        images = ground_truth_data["images"]
        target_images = {}
        for image in images:
            file_name = image["file_name"]
            full_path = os.path.join(image_dir, file_name)
            if tf.gfile.Exists(full_path):
                target_images[image["id"]] = image

        print('Num of target images : ', len(target_images))

        # Get annotation index form 'annotation' and categorn index
        annotations_index = {}
        for annotation in ground_truth_data["annotations"]:
            if annotation["category_id"] in category_index:
                image_id = annotation["image_id"]
                if image_id in target_images:
                    annotations_index[image_id] = annotation

        print('Num of annotations index : ', len(annotations_index))

        # Create Labels.
        labels = {}
        idx = 0
        for annotation in annotations_index:
            if annotations_index["category_id"] not in labels:
                category_id = annotations_index["category_id"]
                cat = category_index[category_id]
                cat['label_index'] = idx
                labels[category_id] = cat
                idx += 1

        print('Num of labels : ', len(labels))

    return labels


def main(_):
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    if not tf.gfile.IsDirectory(FLAGS.temp_dir):
        tf.gfile.MakeDirs(FLAGS.temp_dir)

    annotation_dir = os.path.join(FLAGS.temp_dir, "annotations")
    if not tf.gfile.IsDirectory(annotation_dir):
        tf.gfile.MakeDirs(annotation_dir)

    image_dir = os.path.join(FLAGS.temp_dir, "images")
    if not tf.gfile.IsDirectory(image_dir):
        tf.gfile.MakeDirs(image_dir)

    dataset_utils.download_and_uncompress_tarball(
        _TRAIN_ASIA_ANNOTATION_URL, annotation_dir
    )
    dataset_utils.download_and_uncompress_tarball(
        _VAL_ASIA_ANNOTATION_URL, annotation_dir
    )
    dataset_utils.download_and_uncompress_tarball(
        _UN_OBFUSCATED_NAME_URL, annotation_dir
    )
    # dataset_utils.download_and_uncompress_tarball(_TRAIN_ASIA_DATA_URL, image_dir)

    train_output_path = os.path.join(FLAGS.output_dir, "")
    val_output_path = os.path.join(FLAGS.output_dir, "")
    train_annotation_path = os.path.join(annotation_dir, "train2018.json")
    val_annotation_path = os.path.join(annotation_dir, "val2018.json")
    un_obfuscated_name_file = os.path.join(annotation_dir, "categories.json")

    # Create labels.
    category_index = {}
    category_index = _create_label_from_coco_annotations(
        train_annotation_path, un_obfuscated_name_file, image_dir, FLAGS.super_category
    )

    # Create TF-Record train and test:
    _create_tf_record_from_coco_annotations(
        train_annotation_path,
        category_index,
        image_dir,
        train_output_path,
        FLAGS.super_category,
        num_shards=100,
    )
    _create_tf_record_from_coco_annotations(
        val_annotation_path,
        category_index,
        image_dir,
        val_output_path,
        FLAGS.super_category,
        num_shards=100,
    )

    # Finally, write the labels file:
    class_names = []
    category_index = sorted(category_index, key=lambda x: x["label_index"])
    for category in category_index:
        class_names.append(category["name"])
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, FLAGS.output_dir)


if __name__ == "__main__":
    tf.app.run()
