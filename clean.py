#!/usr/bin/env python2
"""
Attempt to filter bbox by [CleanNet](https://github.com/kuanghuei/clean-net)
(DID NOT WORK)
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import sys
sys.path.append('/home/jamiecho/Repos/models/official/resnet')
from imagenet_preprocessing import _mean_image_subtraction, _CHANNEL_MEANS
from functools import partial

def crop_box(img, box):
    h, w = img.shape[:2]
    box = np.clip(box, 0, 1)
    x0,y0,x1,y1 = np.multiply(box, [w,h,w,h]).astype(np.int32)
    return img[y0:y1,x0:x1]

def write_tsv(extract_fn, X, out_file='/tmp/out.tsv'):
    # sample_key, class name, verification label, [feats...]
    with open(out_file, 'a') as f:
        for k, v in X.iteritems():
            filename = k
            imgs, boxs = v
            if len(imgs) <= 0:
                continue
            feats = extract_fn(imgs)
            for feat, box in zip(feats, boxs):
                f.write('{}_{}\tdrone\t{}\t'.format(filename, box, 1))
                f.write(','.join([str(e) for e in feat.ravel()]))
                f.write('\n')

def parse_and_crop_record(rec_file, outdir='/tmp/crop', viz=False):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    it = tf.python_io.tf_record_iterator(path=rec_file)
    proto = {
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/filename': tf.FixedLenFeature([], tf.string),
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

    res = {}
    idx = 0
    for rec in it:
        ex  = tf.parse_single_example(rec, proto)
        img = tf.image.decode_jpeg(ex['image/encoded']).numpy()

        xmin = tf.sparse.to_dense(ex['image/object/bbox/xmin']).numpy()
        xmax = tf.sparse.to_dense(ex['image/object/bbox/xmax']).numpy()
        ymin = tf.sparse.to_dense(ex['image/object/bbox/ymin']).numpy()
        ymax = tf.sparse.to_dense(ex['image/object/bbox/ymax']).numpy()

        subimgs = []
        for box in zip(xmin,ymin,xmax,ymax):
            subimgs.append( crop_box(img, box) )
        res[ ex['image/filename'].numpy() ] = (subimgs, zip(xmin, ymin, xmax, ymax))

        if viz:
            cv2.imshow('orig', cv2.resize(img[..., ::-1], (640,480)) )
            for i, sim in enumerate(subimgs):
                cv2.imshow('sub-{}'.format(i), cv2.resize(sim[...,::-1], (640,480) ) )
            k = cv2.waitKey(0)
            if k in [27, ord('q')]:
                break
        if len(res) > 128:
            np.save(os.path.join(outdir, '{}.npy'.format(idx)), res)
            idx += 1
            res = {}
    np.save(os.path.join(outdir, '{}.npy'.format(idx)), res)

def format_image(img):
    # resize
    #if np.ndim(img) >= 4:
    if len(img) > 0 and np.ndim(img[0]) >= 3:
        # list of images
        img = [cv2.resize(e, (224,224)) for e in img] # (?, 224, 224, 3)
    else:
        # single image
        img = [cv2.resize(img, (224,224))] # (1,224,224,3)
    # roll channel: bgr->rgb
    img = [e[...,::-1] for e in img]

    # subtract mean
    return np.subtract(img,  _CHANNEL_MEANS)

def main():
    rec_dir  = '/media/ssd/datasets/drones/drone.record'
    crop_dir = '/tmp/crop'
    out_dir  = '/tmp/out.tsv'

    #rec_dir  = '/media/ssd/datasets/drones/png.record'
    #crop_dir = '/tmp/crop2'
    #out_dir  = '/tmp/out2.tsv'

    if not os.path.exists(crop_dir):
        tf.enable_eager_execution()
        parse_and_crop_record(rec_dir, crop_dir)
    else:
        print('CROP: {} already exists'.format(crop_dir))

    if not os.path.exists(out_dir):
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            with graph.as_default():
                #tf.train.import_meta_graph('/tmp/resnet_v2_fp32_savedmodel_NCHW/1538687196/saved_model.pb',

                #img_t   = tf.placeholder(tf.uint8, [None, None, 3])
                #rsz_t   = tf.image.resize_images(img_t, (224, 224))
                #proc    = _mean_image_subtraction(tf.cast(rsz_t, tf.float32), _CHANNEL_MEANS, 3)
                #input_t = tf.expand_dims(proc, 0) # batched
                img_t  = tf.placeholder(tf.float32, [None, 224, 224, 3])

                net = tf.saved_model.loader.load(sess,
                        [tf.saved_model.tag_constants.SERVING],
                        '/tmp/resnet_v2_fp32_savedmodel_NCHW/1538687196/',
                        input_map={'input_tensor:0' : img_t}
                        )
            #writer = tf.summary.FileWriter('/tmp/graphviz', graph)
            #sess.run([])
            #writer.close()
            #print graph.get_operations()
            feat = graph.get_tensor_by_name('resnet_model/Squeeze:0')
            #print 'net', net
            #print type(net), dir(net)
            #print dict(net.signature_def)
            #img_t = graph.get_tensor_by_name('input_tensor:0')
            #img_t = net.signature_def['serving_default'].inputs['input']
            fs = os.listdir(crop_dir)
            extract = lambda img: sess.run(feat, {img_t : format_image(img)})
            for f in fs:
                raw_data = np.load(os.path.join(crop_dir, f)).item()
                write_tsv(extract, raw_data, out_file=out_dir)
    else:
        print('TSV: {} already exists'.format(out_dir) )

if __name__ == '__main__':
    main()
