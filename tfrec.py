#!/usr/bin/env python2
"""
Convert Custom Annotations To Standardized TFRecords.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
#import pandas as pd
import numpy as np
import tensorflow as tf
import sys

sys.path.append("/home/jamiecho/Repos/models/research")
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

#flags = tf.app.flags
#flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
#flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
#flags.DEFINE_string('label', '', 'Name of class label')
## if your image has more labels input them as
## flags.DEFINE_string('label0', '', 'Name of class[0] label')
## flags.DEFINE_string('label1', '', 'Name of class[1] label')
## and so on.
#flags.DEFINE_string('img_path', '', 'Path to images')
#FLAGS = flags.FLAGS

# TO-DO replace this with label map
# for multiple labels add more else if statements
def class_text_to_int(row_label):
    return int(row_label)

    #if row_label == FLAGS.label:  # 'ship':
    #    return 1
    ## comment upper if statement and uncomment these statements for multiple labelling
    ## if row_label == FLAGS.label0:
    ##   return 1
    ## elif row_label == FLAGS.label1:
    ##   return 0
    #else:
    #    None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def to_tf(img_file, boxs):
    with tf.gfile.GFile(img_file, 'rb') as fid:
        encoded_jpg = fid.read()

    # parse metadata
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = img_file.encode('utf8')
    image_format = b'jpg'

    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for box in boxs:
        cx, cy, w, h = box[:4]
        xmins.append(cx - (w/2.0))
        xmaxs.append(cx + (w/2.0))
        ymins.append(cy - (h/2.0))
        ymaxs.append(cy + (h/2.0))
        classes_text.append('drone')
        classes.append( 1 )
        #classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main():
    #ann_folder  = '/media/ssd/datasets/drones/ann/'

    #ann_folder  = '/media/ssd/datasets/drones/ann_proc/'
    #output_path = '/media/ssd/datasets/drones/drone.record'

    #ann_folder  = '/media/ssd/datasets/drones/drone-net-ann/'
    #output_path = '/media/ssd/datasets/drones/drone-net.record'

    ann_folder  = '/tmp/del'
    output_path = '/tmp/png.record'

    writer = tf.python_io.TFRecordWriter(output_path)

    for f_ann in os.listdir(ann_folder):
        ls = open(os.path.join(ann_folder, f_ann)).readlines()
        img_file = ls[0][:-1]
        boxs     = [[float(e) for e in s.split(' ')] for s in ls[2:]]
        try:
            tfrec    = to_tf(img_file, boxs)
            writer.write( tfrec.SerializeToString() )
        except Exception as e:
            print('e', e)
            continue
    writer.close()

    #input_file = ''
    ##nlabels = 2
    #flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
    #flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
    #flags.DEFINE_string('label', '', 'Name of class label')

    #path   = os.path.join(os.getcwd(), FLAGS.img_path)
    ##examples = pd.read_csv(FLAGS.csv_input)
    #grouped = split(examples, 'filename')
    #for group in grouped:
    #    tf_example = create_tf_example(group, path)
    #    writer.write(tf_example.SerializeToString())

    #writer.close()
    #output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    #print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    #tf.app.run()
    main()
