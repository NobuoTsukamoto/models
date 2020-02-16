# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

r"""Convert the AI-Edge Contest Open Dataset(Object detection) to TFRecord for object_detection.

See: https://signate.jp/dlp/ai-edge-contest-data

"""

import hashlib
import io
import logging
import os
import random
import re
import json

import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('image_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/ai_edge_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_integer('num_shards', 10, 'Number of TFRecord shards')
flags.DEFINE_float('prop', 0.2, 'Number of TFRecord shards')

FLAGS = flags.FLAGS


def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       image_file_name,
                       ignore_difficult_instances=False):
    """ Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by running dataset_util.recursive_parse_xml_to_dict)
      label_map_dict: A map from string label names to integers ids.
      image_subdirectory: String specifying subdirectory within the Pascal dataset directory holding the actual image data.
      ignore_difficult_instances: Whether to skip difficult instances in the dataset  (default: False).

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_path = os.path.join(image_subdirectory, image_file_name)
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = 800
    height = 600

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []

    for obj in data:
        xmin = float(obj['box2d']['x1'])
        xmax = float(obj['box2d']['x2'])
        ymin = float(obj['box2d']['y1'])
        ymax = float(obj['box2d']['y2'])

        if xmin > 800 or xmax > 800 or ymin > 600 or ymax > 600:
            print('Error data', obj, img_path)

        xmins.append(xmin / width)
        ymins.append(ymin / height)
        xmaxs.append(xmax / width)
        ymaxs.append(ymax / height)
        class_name = obj['category']
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])

    feature_dict = {
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(
                    image_file_name.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(
                    image_file_name.encode('utf8')),
                'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
                }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):
    """Creates a TFRecord file from examples.

    Args:
      output_filename: Path to where output file is saved.
      num_shards: Number of shards for output file.
      label_map_dict: The label map dictionary.
      annotations_dir: Directory where annotation files are stored.
      image_dir: Directory where image files are stored.
      examples: Examples to parse and save to tf record.
        smaller file sizes.
    """

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                tf_record_close_stack, output_filename, num_shards)

        for idx, example in enumerate(examples):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples))
            json_path = os.path.join(annotations_dir, example)

            if not os.path.exists(json_path):
                logging.warning('Could not find %s, ignoring example.', json_path)
                print('Could not find ignoring example.', json_path)
                continue

            with open(json_path, 'r') as fid:
                json_data = json.load(fid)

            data = json_data['Labels']
            image_file_name = os.path.splitext(example)[0] + '.jpg'

            try:
                tf_example = dict_to_tf_example(data,
                        label_map_dict,
                        image_dir,
                        image_file_name)
                if tf_example:
                    shard_idx = idx % num_shards
                    output_tfrecords[shard_idx].write(tf_example.SerializeToString())
            except ValueError:
                logging.warning('Invalid example: %s, ignoring.', json_path)


# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):

    logging.info('Reading from dataset.')
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    image_dir = FLAGS.image_dir
    annotations_dir = FLAGS.data_dir

    examples_list = [x for x in os.listdir(annotations_dir) if x.endswith(".json")]
    logging.info('json files : %d', len(examples_list))
    print('json files : ', len(examples_list))

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = int(len(examples_list) * FLAGS.prop)
    num_train = int(0.7 * num_examples)
    num_val = int(0.3 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:num_train+num_val]
    print('train num', len(train_examples), ' val num', len(val_examples))
    logging.info('%d training and %d validation examples.', len(train_examples), len(val_examples))

    train_output_path = os.path.join(FLAGS.output_dir, 'ai_edge_object_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'ai_edge_object_val.record')

    create_tf_record(train_output_path,
            FLAGS.num_shards,
            label_map_dict,
            annotations_dir,
            image_dir,
            train_examples)

    create_tf_record(val_output_path,
            FLAGS.num_shards,
            label_map_dict,
            annotations_dir,
            image_dir,
            val_examples)


if __name__ == '__main__':
    tf.app.run()
