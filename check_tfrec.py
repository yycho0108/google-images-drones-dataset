import tensorflow as tf
import sys
import cv2

tf.enable_eager_execution()
rec = tf.data.TFRecordDataset('/tmp/drone.record')

proto ={
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/source_id': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/text': tf.VarLenFeature(tf.string),
        'image/object/class/label': tf.VarLenFeature(tf.int64)
        }

parse = lambda x : tf.parse_single_example(x, proto)
dataset = rec.map(parse)

for rec in dataset.take(1):
    ks = rec.keys()
    #print [rec[k] for k in ks if k != 'image/encoded']
    jpeg = tf.image.decode_jpeg(rec['image/encoded'])
    cv2.imshow('hwat?', jpeg.numpy())
    cv2.waitKey(0)

    #print repr(rec)
#print dataset[0]
